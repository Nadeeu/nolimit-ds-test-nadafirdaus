import numpy as np
import faiss
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import TfidfVectorizer


CLUSTER_TOPIC_MAP = {
    0: 'Politik & Sosial',
    1: 'Cinta & Romansa',
    2: 'Doa & Harapan',
    3: 'Kehidupan Sehari-hari',
    4: 'Kegelisahan & Refleksi'
}


def find_optimal_k(embeddings: np.ndarray, k_range: range) -> dict:
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels, sample_size=1000, random_state=42)
        results[k] = score
    return results


def fit_kmeans(embeddings: np.ndarray, n_clusters: int) -> KMeans:
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(embeddings)
    return km


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatL2:
    embeddings_f32 = embeddings.astype('float32')
    index = faiss.IndexFlatL2(embeddings_f32.shape[1])
    index.add(embeddings_f32)
    return index


def get_topic_by_nearest(embedding: np.ndarray, index: faiss.IndexFlatL2, df_ref) -> str:
    emb_f32 = embedding.astype('float32')
    _, I = index.search(emb_f32, 1)
    return df_ref.iloc[I[0][0]]['topic']


def get_tfidf_keywords(texts: list, top_n: int = 10) -> list:
    vectorizer = TfidfVectorizer(max_features=top_n, min_df=2)
    vectorizer.fit(texts)
    matrix = vectorizer.transform(texts)
    mean_scores = matrix.mean(axis=0).A1
    top_indices = mean_scores.argsort()[::-1][:top_n]
    return [vectorizer.get_feature_names_out()[j] for j in top_indices]