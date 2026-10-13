import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.openai import OpenAI
from llama_index.core.memory import ChatMemoryBuffer
import openai

# --- Page Setup ---
st.set_page_config(page_title="Covid19 Safety Plan Chatbot", page_icon="💬")
st.header("Covid19 Safety Plan Chatbot 💬 🦠 ⛑️")

# --- API Key Configuration ---
# Safely checks both lowercase and uppercase secret configurations
openai_api_key = (
    st.secrets.get("openai_key") 
    or st.secrets.get("OPENAI_API_KEY") 
)

if not openai_api_key:
    st.error("OpenAI API key not found. Please set `openai_key` or `OPENAI_API_KEY` in your Streamlit Secrets.")
    st.stop()

openai.api_key = openai_api_key

# --- Configure LlamaIndex Global Settings (Replaces ServiceContext) ---
Settings.llm = OpenAI(
    model="gpt-3.5-turbo", 
    temperature=0.5, 
    api_key=openai_api_key,
    system_prompt=(
        "You are an expert on Covid19 safety plans for workplaces, your job is to answer questions related to it. "
        "Assume that all questions are related to Covid19 workplace safety plans. "
        "Your answers should be based on facts, do not hallucinate features."
    )
)
Settings.embed_model = "local:BAAI/bge-small-en-v1.5" # Or default OpenAI embedding

# --- Load Data Cached Resource ---
@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Loading and indexing the Covid19 safety plan doc, hang tight! This should take 1-2 minutes."):
        reader = SimpleDirectoryReader(input_dir="./data", recursive=True)
        docs = reader.load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index

try:
    index = load_data()
except Exception as e:
    st.error(f"Error loading documents from `./data`: {e}")
    st.stop()

# --- Initialize Chat Engine & Memory ---
if "memory" not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer.from_defaults(token_limit=1500)

chat_engine = index.as_chat_engine(
    chat_mode="context",
    memory=st.session_state.memory,
)

# --- Initialize Chat Message History ---
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ask me a question about Covid19 Safety Plan!"}
    ]

# Display prior chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# --- User Input & Response Generation ---
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# If the last message is from the user, generate the assistant's response
if st.session_state.messages[-1]["role"] == "user":
    user_query = st.session_state.messages[-1]["content"]
    with st.chat_message("assistant"):
        with st.spinner("Generating response..."):
            try:
                response = chat_engine.chat(user_query)
                response_text = str(response)
                st.write(response_text)
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"An error occurred during generation: {e}")
