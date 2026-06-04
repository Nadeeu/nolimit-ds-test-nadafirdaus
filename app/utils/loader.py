import yaml
import numpy as np
import faiss
import pandas as pd
import streamlit as st
import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, ArrayDictionary
from Sastrawi.StopWordRemover.StopWordRemover import StopWordRemover


ID2LABEL = {0: "anger", 1: "fear", 2: "happy", 3: "love", 4: "sadness"}
EMOJI_MAP = {
    "anger"  : "😠",
    "fear"   : "😨",
    "happy"  : "😊",
    "love"   : "❤️",
    "sadness": "😢"
}
COLOR_MAP = {
    "anger"  : "red",
    "fear"   : "orange",
    "happy"  : "green",
    "love"   : "pink",
    "sadness": "blue"
}
EXTRA_STOPWORDS = [
    "kamu", "saya", "aku", "ku", "mu", "nya", "kami", "kita", "dia", "mereka",
    "yang", "dan", "di", "ke", "dari", "dengan", "untuk", "atau", "tapi",
    "karena", "kalau", "jika", "agar", "supaya", "namun", "tetapi", "serta",
    "tidak", "tak", "bukan", "mau", "bisa", "ada", "sudah", "belum",
    "akan", "jadi", "terus", "sama", "aja", "juga", "udah", "sih", "deh",
    "dong", "loh", "lah", "kan", "pun", "tuh", "nih",
    "lebih", "sekali", "sangat", "banyak", "semua", "setiap", "para",
    "hari", "ini", "itu", "sini", "sana", "situ", "sekarang", "nanti",
    "url", "pas", "buat", "pakai", "tahu", "memang", "orang", "apa",
    "sendiri", "salah", "baru", "cuma", "teman"
]

@st.cache_resource
def load_config():
    with open("config.yaml", "r") as f:
        return yaml.safe_load(f)


@st.cache_resource
def load_classifier():
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained(config["models"]["classifier"])
    model = AutoModelForSequenceClassification.from_pretrained(config["models"]["classifier"])
    model.eval()
    return tokenizer, model


@st.cache_resource
def load_embedder():
    config = load_config()
    return SentenceTransformer(config["models"]["embedding"])


@st.cache_resource
def load_faiss():
    emb_path = "artifacts/embeddings.npy"

    # Download from HF Hub if not found locally
    if not os.path.exists(emb_path):
        from huggingface_hub import hf_hub_download
        os.makedirs("artifacts", exist_ok=True)
        emb_path = hf_hub_download(
            repo_id="Nadaa9/indobert-emotion-twitter",
            filename="embeddings.npy",
            local_dir="artifacts"
        )

    embeddings = np.load(emb_path).astype("float32")
    df_ref     = pd.read_csv("artifacts/predictions.csv")

    n     = len(df_ref)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings[:n])

    return index, df_ref


@st.cache_resource
def load_stopword_remover():
    factory = StopWordRemoverFactory()
    all_stopwords = factory.get_stop_words() + EXTRA_STOPWORDS
    dictionary = ArrayDictionary(all_stopwords)
    return StopWordRemover(dictionary)


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http[s]?://\S+|www\.\S+", "", text)
    text = re.sub(r"rt\s+@\w+:?", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\[username\]", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def predict(text: str) -> dict:
    import torch

    tokenizer, model = load_classifier()
    embedder         = load_embedder()
    index, df_ref    = load_faiss()
    stopword_remover = load_stopword_remover()

    # Preprocess
    cleaned = clean_text(text)
    final   = stopword_remover.remove(cleaned)

    # Topic via FAISS nearest neighbor
    emb      = embedder.encode([final]).astype("float32")
    _, I     = index.search(emb, 1)
    topic    = df_ref.iloc[I[0][0]]["topic"]

    # Classify emotion
    inputs = tokenizer(
        text, return_tensors="pt",
        truncation=True, padding=True, max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    import torch.nn.functional as F
    probs      = F.softmax(outputs.logits, dim=-1)[0].numpy()
    pred_id    = int(probs.argmax())
    emotion    = ID2LABEL[pred_id]
    confidence = float(probs[pred_id]) * 100
    all_probs  = {ID2LABEL[i]: round(float(probs[i]) * 100, 2) for i in range(5)}

    return {
        "emotion"   : emotion,
        "confidence": confidence,
        "topic"     : topic,
        "all_probs" : all_probs,
        "emoji"     : EMOJI_MAP[emotion]
    }