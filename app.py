import streamlit as st
import subprocess
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
import fitz  # PyMuPDF
import os
import warnings
import json

# Constants
BOOK_DIR = './literature_data'
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-l6-v2"
EMBEDDINGS_CACHE = './CACHE'
INDEX_DIR = "./content/books/faiss_index"
HF_TOKEN = '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'

# Suppress future warnings from the HuggingFace Hub
warnings.filterwarnings("ignore", category=FutureWarning, module='huggingface_hub.file_download')

# Set Hugging Face API token environment variable
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_TOKEN

# Load the Hugging Face model and embeddings
def initialize_llm(temperature):
    return HuggingFaceEndpoint(repo_id=HF_MODEL, temperature=temperature)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, cache_folder=EMBEDDINGS_CACHE)

# PDF Loader class
class PDFLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        documents = []
        with fitz.open(self.file_path) as pdf_document:
            for page_num in range(pdf_document.page_count):
                page = pdf_document.load_page(page_num)
                text = page.get_text("text")
                documents.append(Document(page_content=text, metadata={"page_num": page_num, "filename": self.file_path}))
        return documents

# Load PDF files from a directory
@st.cache_data
def load_pdf_files(directory):
    documents = []
    for filename in os.listdir(directory):
        if filename.endswith(".pdf"):
            file_path = os.path.join(directory, filename)
            loader = PDFLoader(file_path)
            documents.extend(loader.load())
            print(f"Loaded file: {filename}")
    return documents

# Load the documents
documents = load_pdf_files(BOOK_DIR)

# Check if the index meta file exists
INDEX_META_FILE = os.path.join(INDEX_DIR, "index_meta.json")
if os.path.exists(INDEX_META_FILE):
    with open(INDEX_META_FILE, 'r') as f:
        indexed_files = set(json.load(f))
else:
    indexed_files = set()

# Check if the current files are already indexed
current_files = set(os.listdir(BOOK_DIR))
should_rebuild_index = current_files != indexed_files

# Load the vector database
vector_db = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

# Memory
@st.cache_resource
def init_memory():
    return ConversationBufferMemory(
        memory_key='chat_history',
        output_key='answer',  # Explicitly set the output key
        return_messages=True)

memory = init_memory()

##### Streamlit #####
st.title("Machine Learning Research Engine V1.0")

# Slider to select the number of documents to retrieve
num_docs = st.sidebar.slider("Number of documents to retrieve:", 1, 6, 3)

# Slider for the model's temperature (creativity)
temperature = st.sidebar.slider("Model temperature (creativity):", 0.0, 1.0, 0.8)

# Dropdown to select the response length
response_length = st.sidebar.selectbox(
    "Select response length:",
    ["short", "medium-short", "medium-long", "long"],
    index=2
)

# Set response length instruction
if response_length == "short":
    length_instruction = "Keep your answers short and succinct."
elif response_length == "medium-short":
    length_instruction = "Keep your answers relatively short."
elif response_length == "medium-long":
    length_instruction = "You can provide a detailed answer, but keep it concise."
elif response_length == "long":
    length_instruction = "Provide a detailed and thorough answer."

st.markdown("Welcome to the Machine Learning Research Engine. Enter your question and receive context-based answers from the provided PDF documents.")

# Sidebar button to rebuild the index
with st.sidebar:
    if st.button("Rebuild Index"):
        if st.checkbox("I understand this will be time-consuming"):
            with st.spinner("Rebuilding the vector database..."):
                print("Starting to rebuild the vector database...")
                # Initialize and apply the text splitter
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                split_docs = text_splitter.split_documents(documents)

                # Perform vector embeddings and create a new FAISS index
                vector_db = FAISS.from_documents(split_docs, embeddings)
                vector_db.save_local(INDEX_DIR)

                # Save the current file names to the index meta file
                with open(INDEX_META_FILE, 'w') as f:
                    json.dump(list(current_files), f)
                st.success("Rebuilding the vector database completed!")
                print("Vector database rebuilt successfully.")
                should_rebuild_index = False

# Show the status of the index and offer to rebuild if needed
if should_rebuild_index and 'hide_rebuild_warning' not in st.session_state:
    st.warning("The index needs to be rebuilt due to changes in the directory. Please click the 'Rebuild Index' button in the sidebar to start the process.")
    if st.button("Dismiss"):
        st.session_state['hide_rebuild_warning'] = True
        st.rerun()

# Button to save chat history
if st.sidebar.button("Save Chat History"):
    with open("chat_history.json", "w") as f:
        json.dump(st.session_state.messages, f)
    st.sidebar.success("Chat history saved!")

# Button to load chat history
if st.sidebar.button("Load Chat History"):
    if os.path.exists("chat_history.json"):
        with open("chat_history.json", "r") as f:
            st.session_state.messages = json.load(f)
            # Load chat history into memory
            memory.chat_memory.clear()  # Clear existing memory to avoid duplication
            for message in st.session_state.messages:
                if message["role"] == "user":
                    memory.chat_memory.add_user_message(message["content"])
                else:
                    memory.chat_memory.add_ai_message(message["content"])
        st.sidebar.success("Chat history loaded and memory updated!")
    else:
        st.sidebar.error("No chat history file found!")

# Initialise chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    if message["role"] == "user":
        st.chat_message(message["role"], avatar="🧑‍💻").write(message["content"])
    else:
        st.chat_message(message["role"], avatar="🤖").write(message["content"])

# React to user input
if prompt := st.chat_input("Enter your question:"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Begin spinner before answering question so it's there for the duration
    with st.spinner("Retrieving answer..."):
        # Initialize the model with the selected temperature
        llm = initialize_llm(temperature)

        # Prompt
        template = f"""You are a nice chatbot having a conversation with a human. Answer the question based on the context and previous conversation, but feel free to provide additional information if needed. {length_instruction}

        Previous conversation:
        {{chat_history}}

        Context to answer question:
        {{context}}

        New human question: {{question}}
        Response:"""

        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])

        # Chain
        chain = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=vector_db.as_retriever(search_kwargs={"k": num_docs}),
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": prompt_template}
        )

        # Send question to chain to get answer
        answer = chain({"question": prompt})

        # Extract answer from dictionary returned by chain
        response = answer["answer"]

        # Display chatbot response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
            # Add a button to read the response aloud
            if st.button("Read Aloud", key=f"read_aloud_{len(st.session_state.messages)}"):
                subprocess.run(["say", response])

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Display source documents in the sidebar
        with st.sidebar:
            st.subheader("Context from Retrieved Documents")
            for doc in answer['source_documents']:
                filename = doc.metadata.get("filename", "Unknown file")
                st.write(f"**Page {doc.metadata['page_num']} from {filename}:**")
                st.write(doc.page_content[:200] + "...")