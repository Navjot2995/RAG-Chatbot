import os
import time
from datetime import datetime
from typing import List, Optional, Tuple
import streamlit as st
import streamlit.components.v1 as components  # Required for lottie workaround
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import requests
import markdown2
from dotenv import load_dotenv

# Lottie workaround
def st_lottie(*args, **kwargs):
    try:
        from streamlit_lottie import st_lottie as _st_lottie
        return _st_lottie(*args, **kwargs)
    except:
        return None

# Load environment variables
load_dotenv()

# Constants
LOTTIE_URL = "https://lottie.host/f1216edb-4e09-46e5-8f1e-90c367b6fc13/iM4N0EXuvy.json"
TEMP_FOLDER = "./uploaded_docs"
SUPPORTED_FILE_TYPES = {
    "pdf": PyPDFLoader,
    "docx": Docx2txtLoader,
    "txt": TextLoader
}

# [Rest of your existing code remains the same...]
# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def load_lottie(url: str) -> Optional[dict]:
    """Load Lottie animation from URL"""
    try:
        res = requests.get(url, timeout=10)
        return res.json() if res.status_code == 200 else None
    except Exception:
        return None

def get_secret(key: str) -> str:
    """Get secret from environment variables or Streamlit secrets"""
    return os.getenv(key) or st.secrets.get(key, "")

def initialize_session_state():
    """Initialize all session state variables"""
    defaults = {
        "chat_history": [],
        "vectors": None,
        "embeddings": None,
        "processing_docs": False,
        "groq_api_key": "",
        "model_name": "mixtral-8x7b-32768",
        "temperature": 0.3,
        "document_names": []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# --- Document Processing ---
def process_uploaded_files(uploaded_files: List) -> bool:
    """Process uploaded files and create vector store using Chroma"""
    try:
        st.session_state.processing_docs = True
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        os.makedirs(TEMP_FOLDER, exist_ok=True)
        docs = []
        st.session_state.document_names = []
        
        with st.spinner("Processing documents..."):
            for uploaded_file in uploaded_files:
                file_path = os.path.join(TEMP_FOLDER, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                file_ext = uploaded_file.name.split(".")[-1].lower()
                if file_ext in SUPPORTED_FILE_TYPES:
                    loader = SUPPORTED_FILE_TYPES[file_ext](file_path)
                    docs.extend(loader.load())
                    st.session_state.document_names.append(uploaded_file.name)
            
            if not docs:
                st.error("No text could be extracted from the documents")
                return False
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            final_documents = text_splitter.split_documents(docs)
            
            # Using Chroma instead of FAISS
            st.session_state.vectors = Chroma.from_documents(
                documents=final_documents,
                embedding=st.session_state.embeddings,
                persist_directory=TEMP_FOLDER
            )
            return True
            
    except Exception as e:
        st.error(f"Document processing failed: {str(e)}")
        return False
    finally:
        st.session_state.processing_docs = False

# --- Chat Functionality ---
def generate_response_streaming(user_question: str):
    """Generate streaming AI response using RAG pipeline"""
    try:
        llm = ChatGroq(
            groq_api_key=st.session_state.groq_api_key,
            model_name=st.session_state.model_name,
            temperature=st.session_state.temperature,
            streaming=True
        )
        
        prompt = ChatPromptTemplate.from_template("""
        You are an expert assistant analyzing documents. Use the following context to answer the question.
        Context: {context}
        Question: {input}
        
        Provide a detailed, accurate response with:
        - Proper markdown formatting
        - Bullet points for lists
        - Code blocks when appropriate
        - Citations from the documents when possible
        """)
        
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        response = retrieval_chain.stream({"input": user_question})
        return response
        
    except Exception as e:
        st.error(f"AI response generation failed: {str(e)}")
        return None

# --- UI Components ---
def render_chat_message(role: str, content: str, timestamp: str = None):
    """Render a chat message with proper styling"""
    with st.chat_message(role):
        st.markdown(content, unsafe_allow_html=True)
        if timestamp:
            st.caption(timestamp)

def render_chat_history():
    """Display the chat conversation history"""
    for idx, (q, a, t) in enumerate(st.session_state.chat_history):
        render_chat_message("user", q, t)
        render_chat_message("assistant", markdown2.markdown(a), t)
        if idx < len(st.session_state.chat_history) - 1:
            st.divider()

# --- Main App ---
def main():
    # Initialize app
    st.set_page_config(
        page_title="Advanced RAG Chatbot",
        layout="wide",
        page_icon="🤖",
        initial_sidebar_state="expanded"
    )
    
    # Apply custom CSS
    st.markdown("""
    <style>
        .stApp { background-color: #0f172a; color: white; }
        [data-testid="stSidebar"] > div:first-child { 
            background-color: #0f172a !important; 
            border-right: 1px solid #1e293b;
        }
        .stChatMessage { padding: 12px 16px; border-radius: 12px; margin: 8px 0; }
        .stChatMessage.user { background-color: #3b82f6; color: white; margin-left: auto; }
        .stChatMessage.assistant { background-color: #1e293b; border: 1px solid #334155; }
        .stButton>button { background-color: #3b82f6 !important; border: none !important; }
        .stTextInput>div>div>input { background-color: #1e293b !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)
    
    initialize_session_state()
    
    # Load animation
    animation = load_lottie(LOTTIE_URL)
    
    # Sidebar Configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # API Settings
        with st.expander("🔑 API & Model Settings", expanded=True):
            st.session_state.groq_api_key = st.text_input(
                "Groq API Key",
                type="password",
                value=st.session_state.groq_api_key,
                help="Get your key from https://console.groq.com/keys"
            )
            
            st.session_state.model_name = st.selectbox(
                "Model",
                ["mixtral-8x7b-32768", "llama2-70b-4096", "gemma-7b-it"],
                index=0
            )
            
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.1
            )
        
        # Document Upload
        with st.expander("📄 Document Upload", expanded=True):
            uploaded_files = st.file_uploader(
                "Upload Documents",
                type=list(SUPPORTED_FILE_TYPES.keys()),
                accept_multiple_files=True,
                disabled=st.session_state.processing_docs
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button(
                    "Process Documents",
                    disabled=not uploaded_files or st.session_state.processing_docs,
                    key="process_button"
                ) and uploaded_files:
                    if process_uploaded_files(uploaded_files):
                        st.success(f"Processed {len(st.session_state.document_names)} documents!")
            
            with col2:
                if st.button(
                    "Clear Documents",
                    type="secondary",
                    disabled=st.session_state.processing_docs,
                    key="clear_button"
                ):
                    st.session_state.vectors = None
                    st.session_state.document_names = []
                    if os.path.exists(TEMP_FOLDER):
                        for f in os.listdir(TEMP_FOLDER):
                            os.remove(os.path.join(TEMP_FOLDER, f))
                    st.rerun()
            
            if st.session_state.document_names:
                st.markdown("**Loaded Documents:**")
                for doc in st.session_state.document_names:
                    st.markdown(f"- {doc}")
    
    # Main Chat Interface
    st.title("🤖 Advanced RAG Chatbot")
    st.caption("Chat with your documents using Groq's ultra-fast LLMs")
    
    if animation:
        st_lottie(animation, speed=1, height=200)
    
    # Display chat history
    render_chat_history()
    
    # User input
    if prompt := st.chat_input("Ask about your documents..."):
        if not st.session_state.groq_api_key:
            st.warning("Please enter your Groq API key in the sidebar")
        elif not st.session_state.vectors:
            st.warning("Please upload and process documents first")
        else:
            # Add user question to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.chat_history.append((prompt, "", timestamp))
            st.rerun()
            
            # Generate and stream response
            with st.spinner("Generating response..."):
                response_container = st.empty()
                full_response = ""
                response_stream = generate_response_streaming(prompt)
                
                if response_stream:
                    for chunk in response_stream:
                        if "answer" in chunk:
                            full_response += chunk["answer"]
                            response_container.markdown(
                                markdown2.markdown(full_response), 
                                unsafe_allow_html=True
                            )
                    
                    # Update chat history with complete response
                    st.session_state.chat_history[-1] = (
                        st.session_state.chat_history[-1][0],
                        full_response,
                        st.session_state.chat_history[-1][2]
                    )
                else:
                    st.error("Failed to generate response")

if __name__ == "__main__":
    main()
