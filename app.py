import streamlit as st
from langchain_openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# ---- Setup ----
st.set_page_config(page_title="Kotaro Amon - TerraLens Bot", layout="centered")
st.title("🏡 Kotaro Amon - TerraLens AI Assistant")

# Placeholder knowledge base (replace with scraped TerraLens data)
documents = [
    ("TerraLens is a Real Estate Investment company specializing in Bangalore, Pune, and Mumbai."),
    ("We help investors find profitable residential and commercial properties."),
    ("Contact us at rinzler208@gmail.com or call +91 78230 10892."),
    ("We offer property consultation and appointment booking for site visits."),
]

# Embedding & Vector Store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(documents, embeddings)
retriever = vectorstore.as_retriever()

qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0),
    retriever=retriever,
)

# ---- Chat UI ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.write("👋 Hi there! Welcome to TerraLens. Ask me anything about real estate investments.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Type your question..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    result = qa({"question": user_input, "chat_history": st.session_state.messages})
    bot_reply = result["answer"]

    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)

