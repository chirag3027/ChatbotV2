import streamlit as st
import os
import json
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

# --- Static prompt for prompt engineering ---
static_prompt = (
    "You are a helpful assistant for a sales team. "
    "Respond in a simple, clear way. Use friendly language."
)

# --- Google Sheets setup ---
def connect_to_gsheet():
    scope = ['https://spreadsheets.google.com/feeds','https://www.googleapis.com/auth/drive']
    creds_dict = json.loads(st.secrets["GSERVICE_ACCOUNT"],strict=False)
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
    client = gspread.authorize(creds)
    sheet = client.open("Chatbot Feedback").sheet1
    return sheet

# --- Load documents from uploaded files ---
def load_documents(files):
    docs = []
    for file in files:
        path = os.path.join("/tmp", file.name)
        with open(path, "wb") as f:
            f.write(file.read())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(path)
        else:
            st.warning(f"Unsupported file type: {file.name}")
            continue
        docs.extend(loader.load())
    return docs

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of dicts {question, response, feedback}
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "user_input" not in st.session_state:
    st.session_state.user_input = ""

# --- Page UI ---
st.title("📊 Smart Sales Chatbot with Feedback")

# OpenAI API key input
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password") or st.secrets.get("OPENAI_API_KEY")

# Upload multiple files
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT):", type=["pdf","docx","txt"], accept_multiple_files=True)

# Process documents button
if openai_api_key and uploaded_files and st.button("Process Documents"):
    raw_docs = load_documents(uploaded_files)
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(raw_docs)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = FAISS.from_documents(docs, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    st.session_state.qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=False
    )
    st.success("✅ Documents processed. Ask a question below.")

# Connect Google Sheet
sheet = None
try:
    sheet = connect_to_gsheet()
except Exception as e:
    st.warning("Google Sheet not connected (missing or invalid credentials).")

# --- Display chat history ---
st.markdown("### 🧾 Conversation History")
for i, entry in enumerate(st.session_state.chat_history):
    st.markdown(f"**You:** {entry['question']}")
    st.markdown(f"**Bot:** {entry['response']}")

    # Feedback buttons only if no feedback yet
    if entry["feedback"] is None:
        cols = st.columns(2)
        if cols[0].button(f"👍 Helpful", key=f"up-{i}"):
            st.session_state.chat_history[i]["feedback"] = "👍"
            if sheet:
                sheet.append_row([entry["question"], entry["response"], "👍"])
            st.rerun()
        if cols[1].button(f"👎 Not Helpful", key=f"down-{i}"):
            st.session_state.chat_history[i]["feedback"] = "👎"
            if sheet:
                sheet.append_row([entry["question"], entry["response"], "👎"])
            st.rerun()

# --- Show follow-up suggestions only if at least one Q/A ---
if st.session_state.chat_history:
    last_question = st.session_state.chat_history[-1]["question"]
    suggestions = {
        "Give more details": "Can you expand with more details?",
        "Provide examples": "Can you give specific examples?",
        "Show data points from the text": "Can you provide data points from the documents?"
    }
    st.markdown("#### 💡 Try a follow-up:")
    cols = st.columns(len(suggestions))
    for i, (label, addition) in enumerate(suggestions.items()):
        if cols[i].button(label):
            follow_up = f"{last_question}. {addition}"
            full_input = static_prompt + "\n\nUser question: " + follow_up
            response = st.session_state.qa_chain.run(full_input)
            st.session_state.chat_history.append({"question": follow_up, "response": response, "feedback": None})
            st.rerun()

# --- User input fixed at bottom ---
# Use st.empty to place input at bottom of page
input_container = st.empty()

def submit_question():
    if st.session_state.user_input and st.session_state.qa_chain:
        full_input = static_prompt + "\n\nUser question: " + st.session_state.user_input
        response = st.session_state.qa_chain.run(full_input)
        st.session_state.chat_history.append({"question": st.session_state.user_input, "response": response, "feedback": None})
        st.session_state.user_input = ""
        st.rerun()

with input_container.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("Ask your question:", key="user_input", placeholder="Type your question here...")
    submitted = st.form_submit_button("Send", on_click=submit_question)

# --- Inject JS to auto scroll down ---
st.markdown(
    """
    <script>
    const chatBox = window.parent.document.querySelector('section.main > div[data-testid="stAppViewContainer"] > div:nth-child(2)');
    if(chatBox) {
        chatBox.scrollTop = chatBox.scrollHeight;
    }
    </script>
    """,
    unsafe_allow_html=True,
)


