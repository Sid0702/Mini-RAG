# Mini RAG Assistant

A retrieval-augmented generation (RAG) system that answers questions based on your uploaded documents, built with AutoGen.

## Features

- **Document Processing**: Upload up to 5 .txt or .md files
- **Smart Chunking**: Documents are automatically split into optimal chunks
- **Vector Embeddings**: Uses sentence-transformers to create vector embeddings
- **Semantic Search**: FAISS vector database for fast similarity search
- **Auto-Agent Collaboration**: Multiple AutoGen agents work together to analyze and answer questions
- **Source Grounding**: See exactly which parts of your documents were used
- **Interaction Logging**: All questions and answers logged with timestamps

## Setup Instructions

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/Sid0702/Mini-RAG.git
   cd mini-rag-assistant
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up your API key:
   - Option 1: Create a `.env` file based on `.env.example`
   - Option 2: Edit the `OAI_CONFIG_LIST.json` file with your API key

### Running the Application

Start the Streamlit app:
```
streamlit run ui.py
```

Access the application in your browser at `http://localhost:8501`

## Project Structure

- `app.py`: Core functionality for document processing and question answering
- `ui.py`: Streamlit user interface
- `OAI_CONFIG_LIST.json`: Configuration for OpenAI API
- `.env.example`: Example environment variables file
- `requirements.txt`: Dependencies needed to run the application

## How It Works

This Mini RAG Assistant uses a collaborative agent approach with AutoGen:

1. **Document Processing**:
   - Documents are chunked into ~250 token segments
   - Sentence-transformers convert chunks to vector embeddings
   - FAISS index stores vectors for fast retrieval

2. **Question Answering Pipeline**:
   - Question Analyzer: Breaks down the user's question into sub-questions
   - Information Extractor: Retrieves relevant information from the document collection
   - Answer Generator: Creates a comprehensive answer based on the retrieved information

3. **Logging**:
   - All interactions are logged to both JSON and CSV files
   - Each log entry includes timestamp, question, answer, and source documents

## How to Use

1. **Upload Documents**:
   - Click "Browse files" and select up to 5 .txt or .md files
   - Click "Process Documents" to analyze them

2. **Ask Questions**:
   - Type your question in the text field
   - Click "Get Answer" to receive a response
   - View sources used to generate the answer

3. **View Logs**:
   - Expand the "View Interaction Logs" section to see previous questions and answers