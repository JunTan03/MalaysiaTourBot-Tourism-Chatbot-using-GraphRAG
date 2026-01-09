# MalaysiaTourBot: Tourism Chatbot using GraphRAG

> **Note**: This is a **University Group Project** developed for the **TNL6323 (Natural Language Programming)** course at Multimedia University (MMU).

**MalaysiaTourBot** is an intelligent AI-powered chatbot designed to assist users in exploring Malaysia through natural language conversations. By USING **GraphRAG**, the system combines a **Neo4j Knowledge Graph** with semantic search and Large Language Models (LLMs) to provide contextually accurate, grounded, and engaging travel assistance.

## Features
* **GraphRAG Implementation**: Combines structured facts from Neo4j with semantic vector search to reduce hallucinations and provide factual answers.
* **Multilingual Intelligence**: Automatically detects and responds in **English** or **Malay** based on the user's input language.
* **Hybrid Retrieval**: Utilises `BAAI/bge-m3` embeddings for high-density semantic retrieval across multiple indices (Place, State, Type, and Content).
* **Dynamic Visuals**: The UI automatically renders images of tourist attractions mentioned in the bot's response.
* **Conversational Memory**: Tracks past interactions within the session to maintain context during the chat.
* **Safety Guardrails**: Built-in instructions to decline harmful content and stay focused specifically on Malaysian tourism.

## Tech Stack
| Component         | Technology                               |
| :---              | :---                                     |
| **Framework**     | Flask                                    |
| **Database**      | Neo4j Graph Database                     |
| **LLM Inference** | Groq (Llama-4-Maverick-17B-Instruct)     |
| **Embeddings**    | BAAI/bge-m3 (Sentence-Transformers)      |
| **Evaluation**    | RAGAS (Faithfulness, Relevance, Context) |

## Project Structure
* `app.py`: The core Flask application logic and GraphRAG pipeline.
* `KG_construction.ipynb`: Jupyter notebook for building the Neo4j graph and generating vector embeddings.
* `Preprocessing.ipynb`: Script for cleaning raw JSON data and stripping citations.
* `MalaysiaTourBot_Testing.ipynb`: Evaluation suite using the RAGAS framework.
* `tourism_data.json`: The structured dataset containing Malaysian attraction details.
* `chatbot.html`: The frontend user interface.

## Installation & Setup
### 1. Prerequisites
Ensure you have Python 3.8+ installed and a running Neo4j instance (local or AuraDB).

### 2. Clone the Repository
```bash
git clone [https://github.com/JunTan03/MalaysiaTourBot-Tourism-Chatbot-using-GraphRAG.git](https://github.com/JunTan03/MalaysiaTourBot-Tourism-Chatbot-using-GraphRAG.git)
cd MalaysiaTourBot
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
An .env.example file is provided in the root. Rename it to .env and add your actual API credentials.

### 5. Running the App
```bash
python app.py
```
Access the chatbot at http://127.0.0.1:5000/.

## Configuration & Model Updates
Because Groq may occasionally remove or deprecate models (especially "Preview" versions), you may need to update the model ID in the future.

**How to change the model**
1. Check the [Groq Supported Models](https://console.groq.com/docs/models) page for the latest IDs.
2. In `app.py`, locate the `groq_client.chat.completions.create` function and update the `model` parameter.
3. **Recommended Method**: Add a `GROQ_MODEL_ID` variable to your `.env` file and update your code to use `os.getenv("GROQ_MODEL_ID")` so you can switch models without editing the script.

## License
This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

## Contributors
* Multimedia University (MMU) - Faculty of Information Science and Technology
* Ho Jun Wei, How Shue Kei, Koay Xin Kuang, Tan Jun Chen