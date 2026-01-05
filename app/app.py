from flask import Flask, render_template, request, jsonify, session
from flask_session import Session
import os
import torch
from sentence_transformers import SentenceTransformer
from neo4j import GraphDatabase
from groq import Groq
from dotenv import load_dotenv

# Load the variables from .env
load_dotenv()

# Replace hardcoded strings with os.getenv()
URI = os.getenv("NEO4J_URI")
USERNAME = os.getenv("NEO4J_USERNAME")
PASSWORD = os.getenv("NEO4J_PASSWORD")
AUTH = (USERNAME, PASSWORD)

# Set Groq API key
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Number of items to retrieve/display for results and related content (Top k, you may decide based on your needs)
ITEM = 5

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY")
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Create Neo4j driver instance for database interactions
driver = GraphDatabase.driver(URI, auth=AUTH)
groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SentenceTransformer("BAAI/bge-m3").to(device)

# Generate vector embedding from input text using the sentence transformer
def generate_text_embedding(text):
    return model.encode(text, convert_to_numpy=True).tolist()

# Query Neo4j for top-k most similar nodes across multiple label indices
def retrieve_top_k_nodes(query, top_k=ITEM):
    embedding = generate_text_embedding(query)
    results = []

    index_map = {
        "Place": "place_index",
        "Content": "content_index",
        "Type": "type_index",
        "State": "state_index"
    }

    with driver.session() as session:
        for label, index_name in index_map.items():
            res = session.run("""
                CALL db.index.vector.queryNodes($index_name, 3, $embedding)
                YIELD node, score
                RETURN $label AS source, node, score, elementId(node) AS node_id
            """, label=label, index_name=index_name, embedding=embedding)

            for r in res:
                data = r.data()
                node = data["node"]
                node["__id"] = data["node_id"]
                results.append({
                    "source": data["source"],
                    "score": data["score"],
                    "node": node
                })

    results.sort(key=lambda x: -x["score"])
    return results[:top_k]

# Given a Place node id, retrieve detailed info including related types, states, and contents
def expand_place_node(place_id):
    with driver.session() as session:
        res = session.run("""
            MATCH (p:Place)
            WHERE elementId(p) = $id
            OPTIONAL MATCH (p)-[:HAS_TYPE]->(t:Type)
            OPTIONAL MATCH (p)-[:IN_STATE]->(s:State)
            OPTIONAL MATCH (p)-[:HAS_EN_CONTENT]->(c1:Content)
            OPTIONAL MATCH (p)-[:HAS_MS_CONTENT]->(c2:Content)
            RETURN p.title AS title, t.name AS type, s.name AS state,
                   coalesce(c1.text, c2.text) AS summary,
                   p.image_url AS image_url
        """, id=place_id)

        return [{
            "title": r["title"],
            "type": r.get("type", ""),
            "state": r.get("state", ""),
            "summary": r.get("summary", ""),
            "image_url": r.get("image_url", "")
        } for r in res]

# Expand related nodes (places) from a matched node of label State, Type, or Content
def expand_related_nodes(match, max_places=ITEM):
    label = match["source"]
    node = match["node"]
    node_id = node["__id"]
    places = []

    with driver.session() as session:
        if label == "Place":
            return expand_place_node(node_id)

        # Query templates to find places related to given node by label
        queries = {
            "State": """
                MATCH (s:State)<-[:IN_STATE]-(p:Place)
                WHERE elementId(s) = $id
                RETURN elementId(p) AS pid LIMIT $limit
            """,
            "Type": """
                MATCH (t:Type)<-[:HAS_TYPE]-(p:Place)
                WHERE elementId(t) = $id
                RETURN elementId(p) AS pid LIMIT $limit
            """,
            "Content": """
                MATCH (c:Content)<-[:HAS_EN_CONTENT|HAS_MS_CONTENT]-(p:Place)
                WHERE elementId(c) = $id
                RETURN elementId(p) AS pid LIMIT $limit
            """
        }

        if label not in queries:
            return []

        res = session.run(queries[label], id=node_id, limit=max_places)
        for r in res:
            places.extend(expand_place_node(r["pid"]))

    return places

# Generate an answer by querying graph, expanding context, then calling LLM
def generate_graphrag_answer(query, top_k=ITEM, max_related=ITEM):
    matches = retrieve_top_k_nodes(query, top_k=top_k)

    context_strings = []
    context_places = []  
    related_count = 0

    for m in matches:
        if related_count >= max_related:
            break

        related = expand_related_nodes(m, max_places=max_related - related_count)

        # Expand related nodes and collect summaries for context
        for item in related:
            if related_count >= max_related:
                break

            # Extract summary text with fallback options
            summary = item.get('ms_content') or item.get('en_content') or item.get('summary') or "No summary available."
            summary = summary if isinstance(summary, str) else str(summary)

            # Format context string with title and summary
            context_strings.append(
                f"Title: {item.get('title', 'No title')}\n"
                f"Summary: {summary}\n"
            )
            context_places.append(item)
            related_count += 1
    # Combine all context strings for prompt
    context_block = "\n\n".join(context_strings)

    # Retrieve past chat history from session
    chat_history = session.get("chat_history", [])
    history_prompt = ""
    for turn in chat_history:
        user_msg = turn.get("user", "").strip()
        bot_msg = turn.get("bot", "").strip()
        if user_msg and bot_msg:
            history_prompt += f"User: {user_msg}\nAssistant: {bot_msg}\n"

    prompt = f"""
You are a knowledgeable and friendly assistant specializing in Malaysian tourism.
Your goal is to answer user questions based on the context provided below. Please follow these guidelines carefully:

1. If the user asks in *Malay, respond fully in **Malay*.
2. If the user asks in *English, respond fully in **English*.
3. If the query contains a mix of Malay and English, determine which language dominates:  
   - If *60% or more of the words or sentence structure* are in *English, respond in **English*.  
   - If *60% or more* are in *Malay, respond in **Malay*.
4. If the question is *not related to Malaysian tourism*, politely inform the user that you can only assist with Malaysian tourism topics.
5. Do **NOT** answer any questions that are:
   - harmful
   - sexual
   - offensive
   - unrelated to Malaysian tourism
   Politely decline such questions.
6. If the user expresses negative emotions (e.g., sad, depressed, hopeless), respond with gentle encouragement and recommend positive travel destinations to uplift them.

Past Conversation (if any):
{history_prompt}

Context:
{context_block}

Question: {query}

Answer:"""

    # Call Groq API to call the LLM with the generated prompt
    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-maverick-17b-128e-instruct",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_completion_tokens=512,
    )

    raw_answer = completion.choices[0].message.content.strip()
    answer_lower = raw_answer.lower()

    mentioned_places = []
    seen_titles = set()
    for place in context_places:
        title = place.get('title', '').lower()
        if title and title in answer_lower and title not in seen_titles:
            mentioned_places.append(place)
            seen_titles.add(title)

    # Build HTML for images related to mentioned places
    image_items = []
    seen_image_urls = set()
    for place in mentioned_places:
        image_url = place.get("image_url", "")
        title = place.get("title", "")
        if image_url and image_url not in seen_image_urls:
            image_items.append(
                f"<div class='answer-image-item' title='{title}'>"
                f"<img src='{image_url}' alt='{title}' />"
                f"</div>"
            )
            seen_image_urls.add(image_url)

    images_html = f"<div class='answer-images-row'>{''.join(image_items)}</div>"

    final_answer = images_html + f"<div class='answer-text'>{raw_answer}</div>"

    return final_answer

@app.route("/")
def index():
    return render_template("chatbot.html")

# Handle user chat input via POST and return chatbot response as JSON
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json.get("message")
    if not user_input:
        return jsonify({"error": "No message provided"}), 400

    # Initialize session memory if not present
    if "chat_history" not in session:
        session["chat_history"] = []

    try:
        # Generate response
        answer = generate_graphrag_answer(user_input)

        # Append to session chat history
        session["chat_history"].append({"user": user_input, "bot": answer})
        session.modified = True 

        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/reset", methods=["POST"])
def reset():
    session.pop("chat_history", None)
    return '', 204

if __name__ == "__main__":
    app.run(debug=True)
