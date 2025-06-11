from langchain_community.chat_message_histories import StreamlitChatMessageHistory
import streamlit as st
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from more_itertools import chunked
from chain import chain

# -----------------------------
# --- Streamlit Page Config ---
# -----------------------------
st.set_page_config(
    page_title="KFUEIT Smart Chatbot",
    page_icon="https://www.pngkey.com/png/detail/773-7739595_kf-khawaja-fareed-university-png.png",
    layout="wide",
)

# ---------------------
# --- Sidebar Setup ---
# ---------------------
st.sidebar.image("https://www.pngkey.com/png/detail/773-7739595_kf-khawaja-fareed-university-png.png", use_container_width=True)
st.sidebar.title("KFUEIT Virtual Assistant")

with st.sidebar.expander("🏛️ University Info"):
    st.markdown("""
    - 🏫 [Departments](https://kfueit.edu.pk/faculties-departments)
    - 🏫[Faculty](https://www.kfueit.edu.pk/faculties)
    - 🗓️ [Academic Calendar](https://kfueit.edu.pk/academic-calendar)
    - 📞 [Contact Info](https://kfueit.edu.pk/contact)
    - 🔗 [Official Website](https://kfueit.edu.pk)
    """)

# --------------------------
# --- Prompt Configuration --
# --------------------------
prompt = ChatPromptTemplate(
    messages=[
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
)

msgs = StreamlitChatMessageHistory(key="langchain_messages")

# -------------------------
# --- Chat UI Heading -----
# -------------------------
st.markdown(
    """
    <h2 style='color: #2c3e50;'>
        🤖 Chatbot For <span style='color:#0e76a8;'>KFUEIT</span>
    </h2>
    """,
    unsafe_allow_html=True
)

if len(msgs.messages) == 0:
    st.markdown(
        """
        <div style='
            background-color: #1e1e2f; 
            border-radius: 15px; 
            padding: 15px; 
            margin: 10px 0; 
            color: #ffffff; 
            max-width: 70%;
            border: 1px solid #333;
        '>
            <b>🤖 KFUEIT Bot:</b> Hello! How can I assist you today?
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------
# --- Chat History Display --
# -------------------------
for msg in msgs.messages:
    with st.chat_message(msg.type, avatar="🧑" if msg.type == "human" else "🤖"):
        st.markdown(
            f"<div style='padding: 10px; border-radius: 15px; background-color: #1e1e2f; color: white; border: 1px solid #333;'>{msg.content}</div>",
            unsafe_allow_html=True
        )

# -------------------------
# --- Chat Input & Response --
# -------------------------
if prompt_input := st.chat_input("Type your question..."):
    with st.chat_message("human", avatar="🧑"):
        st.markdown(
            f"<div style='padding: 10px; border-radius: 15px; background-color: #1e1e2f; color: white; border: 1px solid #333;'>{prompt_input}</div>",
            unsafe_allow_html=True
        )

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        typing_animation = st.empty()
        full_response = ""

        try:
            typing_animation.markdown("<i>🤖 is typing...</i>", unsafe_allow_html=True)

            _chat_history = st.session_state.langchain_messages[1:40]
            _chat_history_tranform = list(
                chunked([msg.content for msg in _chat_history], n=2)
            )

            response = chain.stream(
                {"question": prompt_input, "chat_history": _chat_history_tranform}
            )

            for res in response:
                full_response += res or ""
                message_placeholder.markdown(
                    f"<div style='padding: 10px; border-radius: 15px; background-color: #1e1e2f; color: white; border: 1px solid #333;'>{full_response}</div>",
                    unsafe_allow_html=True
                )

            msgs.add_user_message(prompt_input)
            msgs.add_ai_message(full_response)

        except Exception as e:
            typing_animation.empty()
            st.error(f"An error occurred: {e}")

        typing_animation.empty()

# -------------------------
# --- Theme Styling (Optional) --
# -------------------------
st.markdown("""
<style>
    .stChatMessage { margin-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)



