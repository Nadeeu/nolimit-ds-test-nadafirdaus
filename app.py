import streamlit as st
import torch
import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import faiss
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover

# ── Page config ────────────────────────────────────────────
st.set_page_config(
    page_title="Indonesian Tweet Emotion Analyzer",
    page_icon="🎭",
    layout="wide"
)

# ── Load config ────────────────────────────────────────────
@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)

# ── Load models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    config = load_config()

    # Classifier
    tokenizer = AutoTokenizer.from_pretrained("artifacts/model_output/best_model")
    model = AutoModelForSequenceClassification.from_pretrained("artifacts/model_output/best_model")
    model.eval()

    # Sentence transformer
    embedder = SentenceTransformer(config["models"]["embedding"])

    return tokenizer, model, embedder

# ── Load FAISS + cluster map ───────────────────────────────
@st.cache_resource
def load_cluster_data():
    embeddings = np.load("artifacts/embeddings.npy").astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    df = pd.read_csv("artifacts/predictions.csv")
    return index, df

# ── Preprocessing ──────────────────────────────────────────
@st.cache_resource
def load_stopword_remover():
    extra_stopwords = [
        'kamu', 'saya', 'aku', 'ku', 'mu', 'nya', 'kami', 'kita', 'dia', 'mereka',
        'yang', 'dan', 'di', 'ke', 'dari', 'dengan', 'untuk', 'atau', 'tapi',
        'karena', 'kalau', 'jika', 'agar', 'supaya', 'namun', 'tetapi', 'serta',
        'tidak', 'tak', 'bukan', 'mau', 'bisa', 'ada', 'sudah', 'belum',
        'akan', 'jadi', 'terus', 'sama', 'aja', 'juga', 'udah', 'sih', 'deh',
        'dong', 'loh', 'lah', 'kan', 'pun', 'tuh', 'nih',
        'lebih', 'sekali', 'sangat', 'banyak', 'semua', 'setiap', 'para',
        'hari', 'ini', 'itu', 'sini', 'sana', 'situ', 'sekarang', 'nanti',
        'url', 'pas', 'buat', 'pakai', 'tahu', 'memang', 'orang', 'apa',
        'sudah', 'sendiri', 'salah', 'bisa', 'baru', 'cuma', 'teman'
    ]
    factory = StopWordRemoverFactory()
    all_stopwords = factory.get_stop_words() + extra_stopwords
    dictionary = ArrayDictionary(all_stopwords)
    return StopWordRemover(dictionary)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'rt\s+@\w+:?', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'\[username\]', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_single(text, tokenizer, model, embedder, index, df_ref, stopword_remover):
    id2label = {0: 'anger', 1: 'fear', 2: 'happy', 3: 'love', 4: 'sadness'}
    emotion_emoji = {'anger': '😠', 'fear': '😨', 'happy': '😊', 'love': '❤️', 'sadness': '😢'}

    # Preprocess
    cleaned = clean_text(text)
    words = cleaned.split()
    normalized = ' '.join(words)  # placeholder if no slang dict in app
    final = stopword_remover.remove(normalized)

    # Topic via FAISS nearest neighbor
    emb = embedder.encode([final]).astype('float32')
    _, I = index.search(emb, 1)
    nearest_idx = I[0][0]
    topic = df_ref.iloc[nearest_idx]['topic'] if 'topic' in df_ref.columns else 'Unknown'

    # Emotion classification
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id = int(np.argmax(probs))
    pred_emotion = id2label[pred_id]
    confidence = float(probs[pred_id]) * 100

    all_probs = {id2label[i]: float(probs[i]) * 100 for i in range(5)}

    return pred_emotion, confidence, topic, all_probs, emotion_emoji[pred_emotion]

# ── UI ─────────────────────────────────────────────────────
st.title("🎭 Indonesian Tweet Emotion Analyzer")
st.markdown("Analyze emotions and topics from Indonesian tweets using **IndoBERT** + **Sentence Transformers**.")
st.divider()

tokenizer, model, embedder = load_models()
index, df_ref = load_cluster_data()
stopword_remover = load_stopword_remover()

tab1, tab2 = st.tabs(["💬 Single Text", "📂 Batch CSV"])

# ── Tab 1: Single text ─────────────────────────────────────
with tab1:
    st.subheader("Single Tweet Analysis")
    text_input = st.text_area("Enter an Indonesian tweet:", height=120,
                               placeholder="contoh: Aku sangat senang hari ini!")

    if st.button("Analyze", type="primary"):
        if not text_input.strip():
            st.warning("Please enter a tweet first.")
        else:
            with st.spinner("Analyzing..."):
                emotion, confidence, topic, all_probs, emoji = predict_single(
                    text_input, tokenizer, model, embedder, index, df_ref, stopword_remover
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicted Emotion", f"{emoji} {emotion.capitalize()}")
            with col2:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col3:
                st.metric("Topic Cluster", topic)

            st.markdown("**Emotion Probability Distribution:**")
            prob_df = pd.DataFrame({
                'Emotion': [f"{e.capitalize()}" for e in all_probs.keys()],
                'Probability (%)': [round(v, 2) for v in all_probs.values()]
            }).sort_values('Probability (%)', ascending=False)
            st.bar_chart(prob_df.set_index('Emotion'))

# ── Tab 2: Batch CSV ───────────────────────────────────────
with tab2:
    st.subheader("Batch CSV Analysis")
    st.markdown("Upload a CSV file with a column named `tweet`.")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file:
        df_upload = pd.read_csv(uploaded_file)

        if 'tweet' not in df_upload.columns:
            st.error("CSV must have a column named `tweet`.")
        else:
            st.write(f"Loaded **{len(df_upload)}** tweets. Preview:")
            st.dataframe(df_upload.head())

            if st.button("Analyze All", type="primary"):
                results = []
                progress = st.progress(0)
                status = st.empty()

                for i, row in df_upload.iterrows():
                    emotion, confidence, topic, _, emoji = predict_single(
                        str(row['tweet']), tokenizer, model,
                        embedder, index, df_ref, stopword_remover
                    )
                    results.append({
                        'tweet': row['tweet'],
                        'topic': topic,
                        'predicted_emotion': f"{emoji} {emotion}",
                        'confidence (%)': round(confidence, 2)
                    })
                    progress.progress((i + 1) / len(df_upload))
                    status.text(f"Processing {i + 1}/{len(df_upload)}...")

                progress.empty()
                status.empty()

                df_results = pd.DataFrame(results)
                st.success(f"Done! Analyzed {len(df_results)} tweets.")
                st.dataframe(df_results)

                # Download button
                csv = df_results.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv,
                    file_name="emotion_analysis_results.csv",
                    mime="text/csv"
                )

# ── Footer ─────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray'>Built with IndoBERT + Sentence Transformers · "
    "Dataset: Indonesian Twitter Emotion · nolimit-ds-test-nadafirdaus</div>",
    unsafe_allow_html=True
)