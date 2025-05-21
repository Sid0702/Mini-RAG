import os
import json
import time
import faiss
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Any
import autogen
import csv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader

# Load environment variables
load_dotenv()

# Configuration
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Good balance of performance and speed
MAX_FILES = 5
CHUNK_SIZE = 250  # tokens for chunking
TOP_K_RESULTS = 1  # Number of chunks to retrieve
LOGS_FILE = "rag_logs.json"
CSV_LOGS_FILE = "rag_logs.csv"
VECTOR_STORE_PATH = "./vectorstore"

# Initialize config from json if available, otherwise use environment variables
CONFIG_FILE = "OAI_CONFIG_LIST.json"
if os.path.exists(CONFIG_FILE):
    config_list = autogen.config_list_from_json(CONFIG_FILE)
else:
    # Fallback to environment variable
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        print("API key not found. Please set OPENAI_API_KEY in .env file or create OAI_CONFIG_LIST.json")
        config_list = []
    else:
        config_list = [{"model": "gpt-4.1-mini", "api_key": api_key}]

# Global state variables
embedding_model = SentenceTransformer(EMBEDDING_MODEL)
documents = []
chunks = []
chunk_embeddings = None
index = None
document_lookup = {}  # Map from chunk index to document source


def initialize_logs():
    """Initialize or load existing logs file"""
    if not os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, 'w') as f:
            json.dump([], f)

    # Create CSV file with headers if it doesn't exist
    if not os.path.exists(CSV_LOGS_FILE):
        with open(CSV_LOGS_FILE, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['timestamp', 'question', 'answer', 'sources'])


def log_interaction(question: str, answer: str, sources: List[str]):
    """Log user question and generated answer to JSON and CSV files"""
    log_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "question": question,
        "answer": answer,
        "sources": sources
    }

    # Read existing logs
    with open(LOGS_FILE, 'r') as f:
        logs = json.load(f)

    # Append new log
    logs.append(log_entry)

    # Write updated logs to JSON
    with open(LOGS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

    # Also write to CSV for easier viewing
    with open(CSV_LOGS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            log_entry["timestamp"],
            log_entry["question"],
            log_entry["answer"],
            ", ".join(log_entry["sources"])
        ])


def chunk_text(text: str, source: str) -> List[Dict]:
    """Split text into chunks of approximately CHUNK_SIZE tokens"""
    # Simple chunking by sentences then grouping into chunks of approximately CHUNK_SIZE
    sentences = text.replace('\n', ' ').split('. ')
    sentences = [s + '.' for s in sentences if s]

    chunks_result = []
    current_chunk = ""

    for sentence in sentences:
        # Very rough estimation of tokens (words / 0.75)
        estimated_tokens = len(current_chunk.split()) / 0.75

        if estimated_tokens > CHUNK_SIZE and current_chunk:
            chunks_result.append({"text": current_chunk, "source": source})
            current_chunk = sentence
        else:
            current_chunk += " " + sentence if current_chunk else sentence

    # Add the last chunk if it's not empty
    if current_chunk:
        chunks_result.append({"text": current_chunk, "source": source})

    return chunks_result


def process_file(file_path, file_name):
    """Process a file based on its type"""
    file_ext = os.path.splitext(file_name)[1].lower()

    if file_ext == '.pdf':
        # Use PyPDFLoader for PDF files
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        content = " ".join([page.page_content for page in pages])
    else:
        # Default to text loader for other files
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

    return content


def load_or_create_index():
    """Load existing index from disk or create a new one"""
    global index, chunks, chunk_embeddings, document_lookup

    if os.path.exists(f"{VECTOR_STORE_PATH}/index.faiss") and os.path.exists(f"{VECTOR_STORE_PATH}/metadata.json"):
        # Load existing index
        index = faiss.read_index(f"{VECTOR_STORE_PATH}/index.faiss")

        with open(f"{VECTOR_STORE_PATH}/metadata.json", 'r') as f:
            metadata = json.load(f)
            chunks = metadata["chunks"]
            document_lookup = metadata["document_lookup"]

        # Load embeddings
        chunk_embeddings = np.load(f"{VECTOR_STORE_PATH}/embeddings.npy")

        return True
    else:
        return False


def save_index():
    """Save index, embeddings, and metadata to disk"""
    global index, chunks, chunk_embeddings, document_lookup

    if not os.path.exists(VECTOR_STORE_PATH):
        os.makedirs(VECTOR_STORE_PATH)

    # Save FAISS index
    faiss.write_index(index, f"{VECTOR_STORE_PATH}/index.faiss")

    # Save embeddings
    np.save(f"{VECTOR_STORE_PATH}/embeddings.npy", chunk_embeddings)

    # Save metadata (chunks and document lookup)
    metadata = {
        "chunks": chunks,
        "document_lookup": document_lookup
    }

    with open(f"{VECTOR_STORE_PATH}/metadata.json", 'w') as f:
        json.dump(metadata, f)


def ingest_documents(files):
    """Process uploaded files and create embeddings"""
    global documents, chunks, chunk_embeddings, index, document_lookup

    if len(files) > MAX_FILES:
        return f"Only {MAX_FILES} files allowed. Please reduce the number of files."

    # Try to load existing index
    has_existing_index = load_or_create_index()

    new_chunks = []

    for file in files:
        # Save the file temporarily
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.getvalue())

        # Process the file based on its type
        file_content = process_file(temp_path, file.name)
        documents.append({"filename": file.name, "content": file_content})

        # Chunk the document
        document_chunks = chunk_text(file_content, file.name)
        new_chunks.extend(document_chunks)

        # Clean up
        os.remove(temp_path)

    if not has_existing_index:
        # First-time setup
        chunks = new_chunks
        texts = [chunk["text"] for chunk in chunks]
        chunk_embeddings = embedding_model.encode(texts)

        # Initialize FAISS index
        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings.astype(np.float32))

        # Create lookup
        document_lookup = {i: chunk["source"] for i, chunk in enumerate(chunks)}
    else:
        # Add to existing index
        existing_chunks_count = len(chunks)
        chunks.extend(new_chunks)

        # Create embeddings for new chunks
        new_texts = [chunk["text"] for chunk in new_chunks]
        new_embeddings = embedding_model.encode(new_texts)

        # Update main embeddings array
        chunk_embeddings = np.vstack([chunk_embeddings, new_embeddings])

        # Add to FAISS index
        index.add(new_embeddings.astype(np.float32))

        # Update document lookup
        for i, chunk in enumerate(new_chunks):
            document_lookup[existing_chunks_count + i] = chunk["source"]

    # Save the updated index
    save_index()

    return f"Successfully ingested {len(documents)} documents with {len(chunks)} total chunks."


def retrieve(query: str) -> List[Dict]:
    """Retrieve relevant chunks for a query"""
    global index, chunks

    if not index or len(chunks) == 0:
        return []

    # Convert query to embedding
    query_embedding = embedding_model.encode([query])

    # Search in FAISS index
    D, I = index.search(query_embedding.astype(np.float32), min(TOP_K_RESULTS, len(chunks)))

    # Get the text of retrieved chunks
    retrieved_chunks = []
    for idx in I[0]:
        if idx >= 0 and idx < len(chunks):  # Valid index check
            chunk = chunks[idx]
            retrieved_chunks.append({
                "text": chunk["text"],
                "source": chunk["source"]
            })

    return retrieved_chunks


def get_knowledge(query: str) -> str:
    """Return relevant context for a query - used by AutoGen agents"""
    chunks_result = retrieve(query)
    if not chunks_result:
        return "No relevant information found."

    # Format chunks into a single context string
    context = "\n\n".join([f"[From {chunk['source']}]: {chunk['text']}" for chunk in chunks_result])
    return context


def answer_question(question: str) -> Tuple[str, List[Dict]]:
    """Use AutoGen to generate an answer based on retrieved chunks"""
    global chunks, config_list

    if not chunks:
        return "Please ingest documents first.", []

    if not config_list:
        return "API configuration is missing. Please set up your API key.", []

    # Set up AutoGen agents for RAG
    llm_config = {
        "cache_seed": 42,
        "config_list": config_list,
        "temperature": 0,
    }

    # Retrieve initial chunks
    retrieved_chunks = retrieve(question)
    if not retrieved_chunks:
        return "No relevant information found in the ingested documents.", []

    try:
        # Initialize the agents
        user_proxy = autogen.UserProxyAgent(
            name="user_proxy",
            llm_config=False,
            human_input_mode="NEVER",
            code_execution_config=False
        )

        question_generator = autogen.AssistantAgent(
            name="question_generator",
            llm_config=llm_config,
            system_message="""
            Create at least 8-10 relevant questions to break down the user's question into various aspects.
            You can generate questions like these:
            - What preparations do we need to solve this problem?
            - What are the underlying principles?
            - What are the DO's and DONT's required to prepare the answer?
            - What different patterns, relationships, or principles do you see while solving this?
            - What kind of errors might we face while solving the problem?
            - What other information is still needed to solve the problem?
            """
        )

        planner_agent = autogen.AssistantAgent(
            name="planner_agent",
            llm_config=llm_config,
            system_message="""
            Being part of a groupchat, 
            Other members of the group have access to a large knowledge corpus.
            You need to create a step by step plan of what other members should be looking for in the corpus 
            to gather the correct set of information in order to answer the user's problem.

            RESPOND BACK WITH ALL THE QUESTIONS THAT ARE NEEDED TO BE SEARCHED ON knowledge_corpus,
            as many questions as you can to understand all the nitty gritties of the question
            <Questions to be searched on the knowledge_corpus>
            <Questions to be searched on the knowledge_corpus>
            <Questions to be searched on the knowledge_corpus>
            .....
            """
        )

        information_extractor = autogen.AssistantAgent(
            name="information_extractor",
            llm_config=llm_config,
            system_message="""
            You will search for information in the knowledge base to answer the original question and sub-questions.
            Use the knowledge_corpus function to search for relevant information.
            Collect all relevant information to answer the questions comprehensively.
            """
        )

        executor_agent = autogen.UserProxyAgent(
            name="knowledge_executor",
            llm_config=llm_config,
            human_input_mode="NEVER",
            code_execution_config=False
        )

        answer_generator = autogen.AssistantAgent(
            name="answer_generator",
            llm_config=llm_config,
            system_message="""
            Create a complete response from all the details you have received.
            The response should be explained with every nitty gritty detail.

            For tabular information, make tables to show the response.

            Your response should follow this template:

            ANSWER
            <Comprehensive answer based on the information provided>

            INSIGHTS
            <Key insights based on the user's question and the answer>

            RECOMMENDATIONS
            <Two new questions for further exploration>

            Reference the specific source documents when possible.
            If the available information doesn't fully answer the question, acknowledge the limitations.
            """
        )

        # Register the knowledge retrieval function
        autogen.register_function(
            get_knowledge,
            caller=information_extractor,
            executor=executor_agent,
            name="knowledge_corpus",
            description="This tool searches the document collection to find relevant information"
        )

        # Define the group chat flow
        def speaker_selection_func(last_speaker: Any, groupchat: Any) -> Any:
            messages = groupchat.messages

            if len(messages) == 1:
                return groupchat.agent_by_name("question_generator")
            elif last_speaker.name == "question_generator":
                return groupchat.agent_by_name("planner_agent")
            elif last_speaker.name == "planner_agent":
                return groupchat.agent_by_name("information_extractor")
            elif last_speaker.name == "information_extractor":
                return groupchat.agent_by_name("knowledge_executor")
            elif last_speaker.name == "knowledge_executor":
                return groupchat.agent_by_name("answer_generator")
            else:
                return None  # End conversation

        # Setup group chat
        group_chat = autogen.GroupChat(
            agents=[user_proxy, question_generator, planner_agent, information_extractor, executor_agent,
                    answer_generator],
            messages=[],
            max_round=12,
            speaker_selection_method=speaker_selection_func
        )

        # Create manager
        manager = autogen.GroupChatManager(
            group_chat,
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0
                                         or x.get("name") == "answer_generator",
            llm_config=llm_config
        )

        # Start the conversation
        user_proxy.initiate_chat(
            manager,
            message=f"Question about documents: {question}"
        )

        # Get the final response from the answer generator
        final_message_idx = -1
        for i, msg in enumerate(reversed(group_chat.messages)):
            if msg["name"] == "answer_generator":
                final_message_idx = len(group_chat.messages) - 1 - i
                break

        if final_message_idx >= 0:
            answer = group_chat.messages[final_message_idx]["content"]
        else:
            answer = "Failed to generate a complete answer."

        # Log the interaction
        log_interaction(
            question,
            answer,
            [chunk["source"] for chunk in retrieved_chunks]
        )

        return answer, retrieved_chunks

    except Exception as e:
        error_msg = f"Error generating answer: {str(e)}"
        return error_msg, retrieved_chunks


# Initialize logs
initialize_logs()