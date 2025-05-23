import streamlit as st
import os
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI

# Title
st.title("📄 Document-Based Q&A Chatbot")

# Input OpenAI API Key
openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")

# Upload multiple files
uploaded_files = st.file_uploader("Upload documents (PDF, DOCX, TXT):", type=["pdf", "docx", "txt"], accept_multiple_files=True)

# Input question
user_query = st.text_input("Enter your question:")

# Helper to load and process documents
def load_documents(files):
    docs = []
    for file in files:
        file_path = os.path.join("/tmp", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            st.error(f"Unsupported file type: {file.name}")
            continue

        docs.extend(loader.load())
    return docs

# Process and answer the query
if openai_api_key and uploaded_files and user_query:
    try:
        # Load and chunk documents
        raw_docs = load_documents(uploaded_files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(raw_docs)

        # Create embeddings and vector store
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = FAISS.from_documents(docs, embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 3})

        # QA chain that also returns sources
        llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
        qa_chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever)

        # Run the query
        result = qa_chain({"question": user_query})

        answer = result.get("answer", "")
        sources = result.get("sources", "")

        # Show answer
        if not answer or "I don't know" in answer or len(answer.strip()) < 10:
            st.warning("Good answer not found.")
        else:
            st.success(answer)

            # Optional: show source snippets in dropdown
            with st.expander("🔍 Show source snippets"):
                if sources:
                    st.markdown(sources)
                else:
                    st.markdown("No specific sources provided.")

    except Exception as e:
        st.error(f"Error: {e}")
