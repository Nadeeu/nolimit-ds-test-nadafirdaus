import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import numpy as np
import pandas as pd
import faiss
import streamlit as st
import yaml

from src.preprocessing import clean_tweet, normalize, remove_stopwords, load_kamus, ALL_STOPWORDS
from src.embeddings import load_embedder, load_embeddings
from src.clustering import build_faiss_index, get_topic_by_nearest
from src.classifier import load_model, predict_emotion

EMOJI_MAP = {
    'anger'  : '😠',
    'fear'   : '😨',
    'happy'  : '😊',
    'love'   : '❤️',
    'sadness': '😢'
}

COLOR_MAP = {
    'anger'  : 'red',
    'fear'   : 'orange',
    'happy'  : 'green',
    'love'   : 'pink',
    'sadness': 'blue'
}


@st.cache_resource
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


@st.cache_resource
def get_classifier():
    config = load_config()
    return load_model(config['models']['classifier'])


@st.cache_resource
def get_embedder():
    config = load_config()
    return load_embedder(config['models']['embedding'])


@st.cache_resource
def get_faiss():
    emb_path = 'artifacts/embeddings.npy'

    if not os.path.exists(emb_path):
        from huggingface_hub import hf_hub_download
        os.makedirs('artifacts', exist_ok=True)
        emb_path = hf_hub_download(
            repo_id='Nadaa9/indobert-emotion-twitter',
            filename='embeddings.npy',
            local_dir='artifacts'
        )

    embeddings = load_embeddings(emb_path).astype('float32')
    df_ref     = pd.read_csv('artifacts/predictions.csv')
    index      = build_faiss_index(embeddings[:len(df_ref)])

    return index, df_ref


@st.cache_resource
def get_kamus():
    return load_kamus('data/kamus_singkatan.csv')


def predict(text: str) -> dict:
    tokenizer, model = get_classifier()
    embedder         = get_embedder()
    index, df_ref    = get_faiss()
    kamus_dict       = get_kamus()

    cleaned    = clean_tweet(text)
    normalized = normalize(cleaned, kamus_dict)
    final      = remove_stopwords(normalized)

    emb   = embedder.encode([final]).astype('float32')
    topic = get_topic_by_nearest(emb, index, df_ref)

    result          = predict_emotion(text, tokenizer, model)
    result['topic'] = topic
    result['emoji'] = EMOJI_MAP[result['emotion']]

    return result