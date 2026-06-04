import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.loader import COLOR_MAP, EMOJI_MAP
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Analytics")
st.markdown("Insights from the **Indonesian Twitter Emotion** test set predictions.")
st.divider()

@st.cache_data
def load_predictions():
    return pd.read_csv("artifacts/predictions.csv")

df = load_predictions()


total       = len(df)
accuracy    = (df["predicted_emotion"] == df["true_emotion"]).mean() * 100
top_emotion = df["predicted_emotion"].value_counts().idxmax()
top_topic   = df["topic"].value_counts().idxmax()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Tweets",        f"{total:,}")
k2.metric("Accuracy",            f"{accuracy:.1f}%")
k3.metric("Most Common Emotion", f"{EMOJI_MAP.get(top_emotion, '')} {top_emotion.capitalize()}")
k4.metric("Most Common Topic",   top_topic)
st.divider()


col1, col2 = st.columns(2)

with col1:
    st.subheader("Emotion Distribution")
    emo_counts = df["predicted_emotion"].value_counts().reset_index()
    emo_counts.columns = ["Emotion", "Count"]
    colors = [COLOR_MAP.get(e, "gray") for e in emo_counts["Emotion"]]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(emo_counts["Emotion"], emo_counts["Count"], color=colors)
    ax.set_xlabel("Emotion")
    ax.set_ylabel("Count")
    ax.set_title("Predicted Emotion Distribution")
    plt.tight_layout()
    st.pyplot(fig)

with col2:
    st.subheader("Topic Distribution")
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(topic_counts["Topic"], topic_counts["Count"], color="steelblue")
    ax.set_xlabel("Count")
    ax.set_title("Topic Cluster Distribution")
    plt.tight_layout()
    st.pyplot(fig)

st.divider()


st.subheader("Emotion Distribution per Topic")
crosstab = pd.crosstab(df["topic"], df["predicted_emotion"])

fig, ax = plt.subplots(figsize=(10, 4))
sns.heatmap(crosstab, annot=True, fmt="d", cmap="Blues", ax=ax)
ax.set_title("Emotion per Topic Heatmap")
ax.set_xlabel("Predicted Emotion")
ax.set_ylabel("Topic")
plt.tight_layout()
st.pyplot(fig)

st.divider()


col3, col4 = st.columns(2)

with col3:
    st.subheader("True vs Predicted Emotion")
    true_counts = df["true_emotion"].value_counts().rename("True")
    pred_counts = df["predicted_emotion"].value_counts().rename("Predicted")
    compare_df  = pd.concat([true_counts, pred_counts], axis=1).fillna(0)
    st.bar_chart(compare_df)

with col4:
    st.subheader("Sample Tweets per Emotion")
    selected = st.selectbox("Select emotion:", options=sorted(df["predicted_emotion"].unique()))
    samples  = (
        df[df["predicted_emotion"] == selected]["tweet"]
        .sample(min(5, len(df[df["predicted_emotion"] == selected])), random_state=42)
        .tolist()
    )
    for i, tweet in enumerate(samples, 1):
        st.markdown(f"**{i}.** {tweet}")

st.divider()
st.caption("Model: IndoBERT · Dataset: Indonesian Twitter Emotion · nolimit-ds-test-nadafirdaus")