import os
import yaml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from torch.utils.data import Dataset
import torch

from src.preprocessing import clean_tweet, normalize, remove_stopwords, load_kamus
from src.embeddings import load_embedder, generate_embeddings, save_embeddings
from src.clustering import fit_kmeans, build_faiss_index, get_tfidf_keywords
from src.classifier import load_model, compute_class_weights, WeightedTrainer, predict_emotion, LABEL2ID, ID2LABEL


# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)


# Load data
print("1. LOADING DATA")

df = pd.read_csv(config['paths']['data'])
df = df.drop_duplicates(subset='tweet')
print(f"Loaded {len(df)} tweets")
print(df['label'].value_counts())


# Preprocessing
print("\n2. PREPROCESSING")

kamus_dict = load_kamus(config['paths']['slang_dict'])

df['clean_tweet']  = df['tweet'].apply(clean_tweet)
df['normal_tweet'] = df['clean_tweet'].apply(lambda x: normalize(x, kamus_dict))
df['final_tweet']  = df['normal_tweet'].apply(remove_stopwords)

print("Preprocessing done.")
print(df[['tweet', 'clean_tweet', 'normal_tweet', 'final_tweet']].head(3))

# Embeddings
print("\n3. GENERATING EMBEDDINGS")

os.makedirs('artifacts', exist_ok=True)
emb_path = config['paths']['embeddings']

if os.path.exists(emb_path):
    print("Embeddings already exist, loading from disk...")
    embeddings = np.load(emb_path)
else:
    embedder   = load_embedder(config['models']['embedding'])
    embeddings = generate_embeddings(df['final_tweet'].tolist(), embedder, batch_size=config['embedding']['batch_size'])
    save_embeddings(embeddings, emb_path)

print(f"Embeddings shape: {embeddings.shape}")

# Clustering
print("\n4. CLUSTERING")

n_clusters = config['clustering']['n_clusters']
km         = fit_kmeans(embeddings, n_clusters)
df['cluster'] = km.labels_

print(f"KMeans fitted with k={n_clusters}")
print("Cluster distribution:")
print(df['cluster'].value_counts().sort_index())

print("\nTop TF-IDF keywords per cluster:")
for i in range(n_clusters):
    cluster_tweets = df[df['cluster'] == i]['final_tweet'].tolist()
    keywords       = get_tfidf_keywords(cluster_tweets)
    print(f"  Cluster {i}: {keywords}")

topic_map    = {int(k): v for k, v in config['clustering']['topic_labels'].items()}
df['topic']  = df['cluster'].map(topic_map)
print("\nTopic distribution:")
print(df['topic'].value_counts())


# Faiss Index
print("\n5. BUILDING FAISS INDEX")

index = build_faiss_index(embeddings)
print(f"FAISS index built with {index.ntotal} vectors")

# classifier
print("\n6. TRAINING CLASSIFIER")

df['label'] = df['label'].map(LABEL2ID)

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
test_df,  val_df  = train_test_split(test_df, test_size=0.5, random_state=42, stratify=test_df['label'])

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")


class EmoDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=64):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding=True,
            max_length=max_length, return_tensors='pt'
        )
        self.labels = torch.tensor(list(labels), dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item


tokenizer_train = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
model_train     = AutoModelForSequenceClassification.from_pretrained(
    'indobenchmark/indobert-base-p1',
    num_labels=5,
    id2label=ID2LABEL,
    label2id=LABEL2ID
)

train_dataset = EmoDataset(train_df['normal_tweet'], train_df['label'], tokenizer_train)
val_dataset   = EmoDataset(val_df['normal_tweet'],   val_df['label'],   tokenizer_train)
test_dataset  = EmoDataset(test_df['normal_tweet'],  test_df['label'],  tokenizer_train)

weights_tensor = compute_class_weights(train_df['label'].values)

training_args = TrainingArguments(
    output_dir        = config['paths']['model_output'],
    num_train_epochs  = config['training']['epochs'],
    per_device_train_batch_size = config['training']['batch_size'],
    per_device_eval_batch_size  = config['training']['batch_size'],
    learning_rate     = config['training']['learning_rate'],
    warmup_steps      = config['training']['warmup_steps'],
    weight_decay      = config['training']['weight_decay'],
    eval_strategy     = 'epoch',
    save_strategy     = 'epoch',
    load_best_model_at_end  = True,
    metric_for_best_model   = 'f1_macro',
    logging_dir       = 'artifacts/logs',
    logging_steps     = 50,
    fp16              = torch.cuda.is_available(),
    report_to         = 'none'
)

from sklearn.metrics import accuracy_score, f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1_macro': f1_score(labels, preds, average='macro')
    }

trainer = WeightedTrainer(
    weights_tensor = weights_tensor,
    model          = model_train,
    args           = training_args,
    train_dataset  = train_dataset,
    eval_dataset   = val_dataset,
    compute_metrics= compute_metrics,
)

trainer.train()

# Evaluation
print("\n7. EVALUATION")

predictions = trainer.predict(test_dataset)
preds       = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids

print(classification_report(true_labels, preds, target_names=list(ID2LABEL.values())))


# Saving
print("\n8. SAVING ARTIFACTS")

trainer.save_model(config['paths']['model_output'])
tokenizer_train.save_pretrained(config['paths']['model_output'])

df_test = test_df.copy()
df_test['predicted_emotion'] = [ID2LABEL[p] for p in preds]
df_test['true_emotion']      = [ID2LABEL[t] for t in true_labels]
df_test['topic']             = df_test.index.map(df['topic'])

df_test[['tweet', 'topic', 'predicted_emotion', 'true_emotion']].to_csv(
    'artifacts/predictions.csv', index=False
)

sample = df_test[['tweet', 'topic', 'predicted_emotion', 'true_emotion']].sample(20, random_state=42)
sample.to_csv(config['paths']['sample_data'], index=False)

print("All artifacts saved!")
print(f"  Model       → {config['paths']['model_output']}")
print(f"  Embeddings  → {config['paths']['embeddings']}")
print(f"  Predictions → artifacts/predictions.csv")
print(f"  Sample      → {config['paths']['sample_data']}")
print("\nPipeline complete!")