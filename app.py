import os

import streamlit as st
from groq import Groq

st.set_page_config(page_title="Groq LLM Chat", page_icon="🤖", layout="centered")

FALLBACK_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]


# ---------- API key handling ----------
def get_api_key():
    # 1) Streamlit secrets (Community Cloud), 2) environment variable, 3) sidebar input
    try:
        if "GROQ_API_KEY" in st.secrets:
            return st.secrets["GROQ_API_KEY"]
    except Exception:
        pass
    return os.environ.get("GROQ_API_KEY") or st.session_state.get("manual_key")


@st.cache_data(ttl=3600, show_spinner=False)
def list_models(api_key: str):
    """Fetch currently available chat models from Groq; fall back to a static list."""
    try:
        client = Groq(api_key=api_key)
        ids = sorted(
            m.id
            for m in client.models.list().data
            if not any(x in m.id.lower() for x in ("whisper", "tts", "guard", "embed"))
        )
        return ids or FALLBACK_MODELS
    except Exception:
        return FALLBACK_MODELS


# ---------- Sidebar ----------
with st.sidebar:
    st.title("⚙️ Settings")

    api_key = get_api_key()
    if not api_key:
        manual = st.text_input("Groq API key", type="password", help="Get one free at console.groq.com")
        if manual:
            st.session_state["manual_key"] = manual
            api_key = manual

    models = list_models(api_key) if api_key else FALLBACK_MODELS
    default_idx = models.index("llama-3.3-70b-versatile") if "llama-3.3-70b-versatile" in models else 0
    model = st.selectbox("Model", models, index=default_idx)

    system_prompt = st.text_area(
        "System prompt",
        value="You are a helpful, concise AI assistant.",
        height=100,
    )
    temperature = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
    max_tokens = st.slider("Max tokens", 64, 4096, 1024, 64)

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------- Main UI ----------
st.title("🤖 Groq LLM Chat")
st.caption(f"Powered by Groq · model: `{model}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if not api_key:
    st.info("Enter your Groq API key in the sidebar to start chatting.")
    st.stop()

if prompt := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    client = Groq(api_key=api_key)
    api_messages = [{"role": "system", "content": system_prompt}] + st.session_state.messages

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""
        try:
            stream = client.chat.completions.create(
                model=model,
                messages=api_messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full_response += delta
                placeholder.markdown(full_response + "▌")
            placeholder.markdown(full_response)
        except Exception as e:
            full_response = f"⚠️ Error: {e}"
            placeholder.error(full_response)

    st.session_state.messages.append({"role": "assistant", "content": full_response})
