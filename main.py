import os
import pickle
from dotenv import load_dotenv
import streamlit as st
from groq import Groq
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

# ===================== 🔐 Load Environment ===================== #
load_dotenv()
api_key = os.getenv("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY")

if not api_key:
    st.error("❌ GROQ_API_KEY not found! Add it to your `.env` file or Streamlit Secrets.")
    st.stop()

# ===================== 🧠 Initialize Groq ===================== #
client = Groq(api_key=api_key)

# ===================== 🖥️ Streamlit UI Setup ===================== #
st.set_page_config(page_title="Groq + LangChain QA", layout="centered")
st.title("📰 News Article Q&A with Groq + LangChain")

urls_input = st.text_area("🔗 Enter up to 3 article URLs (one per line):", height=150)
urls = [url.strip() for url in urls_input.split("\n") if url.strip()]

# ===================== 📄 Process & Save Vector Index ===================== #
if st.button("📥 Load & Index Articles"):
    if not urls:
        st.warning("⚠️ Please enter at least one URL.")
    else:
        with st.spinner("Loading and processing articles..."):
            try:
                loader = UnstructuredURLLoader(urls=urls)
                documents = loader.load()
                st.success(f"✅ Loaded {len(documents)} documents")

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                docs = splitter.split_documents(documents)

                embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                vector_index = FAISS.from_documents(docs, embedding_model)

                with open("vector_index.pkl", "wb") as f:
                    pickle.dump(vector_index, f)

                st.success("📦 Vector index created and saved as `vector_index.pkl`")
            except Exception as e:
                st.error(f"🚫 Error: {str(e)}")

# ===================== ❓ Question Answering ===================== #
if os.path.exists("vector_index.pkl"):
    with open("vector_index.pkl", "rb") as f:
        vector_index = pickle.load(f)

    retriever = vector_index.as_retriever()
    query = st.text_input("🧠 Ask a question based on the articles:")

    if st.button("🔎 Get Answer") and query:
        with st.spinner("Asking Groq LLM..."):
            try:
                docs = retriever.invoke(query)
                context = "\n\n".join([doc.page_content for doc in docs])
                sources = [doc.metadata.get("source", "N/A") for doc in docs]

                response = client.chat.completions.create(
                    model="meta-llama/llama-4-scout-17b-16e-instruct",
                    messages=[
                        {
                            "role": "system",
                            "content": "Answer the user's question using only the provided context. Include article sources if relevant."
                        },
                        {
                            "role": "user",
                            "content": f"Context:\n{context}\n\nQuestion: {query}"
                        }
                    ],
                    temperature=0.7,
                    max_completion_tokens=512,
                    stream=False
                )

                st.subheader("📄 Answer:")
                st.write(response.choices[0].message.content.strip())

                st.subheader("🔗 Sources:")
                for src in sources:
                    st.markdown(f"- {src}")

            except Exception as e:
                st.error(f"🚫 Error during answering: {str(e)}")
