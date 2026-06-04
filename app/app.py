import streamlit as st

st.set_page_config(
    page_title="Indonesian Tweet Emotion Analyzer",
    page_icon="🎭",
    layout="wide"
)

st.sidebar.title("🎭 Emotion Analyzer")
st.sidebar.divider()
st.sidebar.caption("nolimit-ds-test-nadafirdaus · Nada Firdaus")

st.title("🎭 Indonesian Tweet Emotion Analyzer")
st.markdown("Analyze emotions and topics from Indonesian tweets using **IndoBERT** + **Sentence Transformers**.")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Model",     "IndoBERT")
col2.metric("Embedding", "MiniLM-L12-v2")
col3.metric("Accuracy",  "71%")
col4.metric("F1 Score",  "0.72")

st.divider()

st.markdown("### 👈 Use the sidebar to navigate")
st.markdown(
    "- **Prediction** — analyze a single tweet or upload a CSV batch\n"
    "- **Analytics** — explore emotion and topic insights from the dataset"
)