import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

EXTRA_STOPWORDS = [
    'kamu', 'saya', 'aku', 'ku', 'mu', 'nya', 'kami', 'kita', 'dia', 'mereka',
    'yang', 'dan', 'di', 'ke', 'dari', 'dengan', 'untuk', 'atau', 'tapi',
    'karena', 'kalau', 'jika', 'agar', 'supaya', 'namun', 'tetapi', 'serta',
    'tidak', 'tak', 'bukan', 'mau', 'bisa', 'ada', 'sudah', 'belum',
    'akan', 'jadi', 'terus', 'sama', 'aja', 'juga', 'udah', 'sih', 'deh',
    'dong', 'loh', 'lah', 'kan', 'pun', 'tuh', 'nih',
    'lebih', 'sekali', 'sangat', 'banyak', 'semua', 'setiap', 'para',
    'hari', 'ini', 'itu', 'sini', 'sana', 'situ', 'sekarang', 'nanti',
    'url', 'pas', 'buat', 'pakai', 'tahu', 'memang', 'orang', 'apa',
    'sendiri', 'salah', 'baru', 'cuma', 'teman'
]

factory = StopWordRemoverFactory()
ALL_STOPWORDS = set(factory.get_stop_words() + EXTRA_STOPWORDS)


def clean_tweet(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http[s]?://\S+|www\.\S+', '', text)
    text = re.sub(r'rt\s+@\w+:?', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'rt\s+\[username\]:?', '', text)
    text = re.sub(r'\[username\]', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()


def normalize(text: str, kamus_dict: dict) -> str:
    words = text.split()
    return ' '.join([kamus_dict.get(w, w) for w in words])


def remove_stopwords(text: str) -> str:
    words = text.split()
    return ' '.join([w for w in words if w not in ALL_STOPWORDS])


def load_kamus(path: str) -> dict:
    df = pd.read_csv(path, sep=';', header=None, names=['singkatan', 'hasil'])
    df['singkatan'] = df['singkatan'].astype(str).str.strip()
    df['hasil'] = df['hasil'].astype(str).str.strip()
    return dict(zip(df['singkatan'], df['hasil']))


def preprocess(text: str, kamus_dict: dict) -> dict:
    cleaned = clean_tweet(text)
    normalized = normalize(cleaned, kamus_dict)
    final = remove_stopwords(normalized)
    return {
        'clean_tweet': cleaned,
        'normal_tweet': normalized,
        'final_tweet': final
    }