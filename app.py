import streamlit as st
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ---- Config ----
st.set_page_config(page_title="Kotaro Amon - TerraLens Bot", layout="centered")
st.title("🏡 Kotaro Amon - TerraLens AI Assistant")

# OpenAI Client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Knowledge Base ----
documents = [
    "TerraLens is a Real Estate Investment company specializing in Bangalore, Pune, and Mumbai.",
    "We help investors find profitable residential and commercial properties.",
    "Contact us at rinzler208@gmail.com or call +91 78230 10892.",
    "We offer property consultation and appointment booking for site visits."
]

# Create embeddings
doc_embeddings = []
for doc in documents:
    emb = client.embeddings.create(model="text-embedding-3-small", input=doc)
    doc_embeddings.append(emb.data[0].embedding)
doc_embeddings = np.array(doc_embeddings)

# ---- Session State ----
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "lead_captured" not in st.session_state:
    st.session_state["lead_captured"] = False
if "user_email" not in st.session_state:
    st.session_state["user_email"] = None
if "user_phone" not in st.session_state:
    st.session_state["user_phone"] = None

# ---- Greeting ----
if not st.session_state["messages"]:
    st.session_state["messages"].append(
        {"role": "assistant", "content": "👋 Hi there! Welcome to TerraLens. I’m Kotaro Amon. How can I help you today?"}
    )

# ---- Show chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ---- Handle input ----
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Lead capture logic
    if not st.session_state["lead_captured"]:
        if st.session_state["user_email"] is None:
            st.session_state["user_email"] = user_input
            reply = "📧 Thanks! Could you also share your phone number so we can follow up?"
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.stop()

        elif st.session_state["user_phone"] is None:
            st.session_state["user_phone"] = user_input
            st.session_state["lead_captured"] = True
            reply = "✅ Got it! Thanks for sharing your details. Now, how can I assist you with real estate?"
            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.markdown(reply)
            st.stop()

    # ---- Knowledge Retrieval ----
    q_emb = client.embeddings.create(model="text-embedding-3-small", input=user_input).data[0].embedding
    sims = cosine_similarity([q_emb], doc_embeddings)[0]
    best_doc = documents[int(np.argmax(sims))]

    # ---- Ask GPT ----
    prompt = f"You are Kotaro Amon, a real estate assistant for TerraLens. Use this context:\n{best_doc}\n\nUser: {user_input}\nAnswer:"
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for TerraLens real estate customers."},
            {"role": "user", "content": prompt}
        ]
    )
    bot_reply = completion.choices[0].message.content

    # ---- Show response ----
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})
    with st.chat_message("assistant"):
        st.markdown(bot_reply)
