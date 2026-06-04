import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.loader import predict, EMOJI_MAP
import streamlit as st
import pandas as pd

st.title("🔍 Prediction")
st.markdown("Detect the **emotion** and **topic** of Indonesian tweets.")
st.divider()

tab1, tab2 = st.tabs(["💬 Single Text", "📂 Batch CSV"])

with tab1:
    st.subheader("Single Tweet Analysis")
    text_input = st.text_area(
        "Enter an Indonesian tweet:",
        height=120,
        placeholder="contoh: Aku sangat senang hari ini!"
    )

    if st.button("Analyze", type="primary", key="single"):
        if not text_input.strip():
            st.warning("Please enter a tweet first.")
        else:
            with st.spinner("Analyzing..."):
                result = predict(text_input)

            col1, col2, col3 = st.columns(3)
            col1.metric(
                "Predicted Emotion",
                f"{result['emoji']} {result['emotion'].capitalize()}"
            )
            col2.metric("Confidence", f"{result['confidence']:.1f}%")
            col3.metric("Topic Cluster", result["topic"])

            st.markdown("**Emotion Probability Distribution:**")
            prob_df = (
                pd.DataFrame.from_dict(
                    result["all_probs"], orient="index", columns=["Probability (%)"]
                )
                .sort_values("Probability (%)", ascending=False)
            )
            st.bar_chart(prob_df)


with tab2:
    st.subheader("Batch CSV Analysis")
    st.markdown("Upload a CSV with a column named `tweet`.")

    # Template download
    template = pd.DataFrame({"tweet": [
        "Aku sangat senang hari ini!",
        "Kenapa semua ini terjadi padaku...",
        "Semoga Indonesia selalu damai dan sejahtera"
    ]})
    st.download_button(
        label="⬇️ Download CSV Template",
        data=template.to_csv(index=False).encode("utf-8"),
        file_name="template.csv",
        mime="text/csv"
    )

    uploaded = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded:
        df_up = pd.read_csv(uploaded)

        if "tweet" not in df_up.columns:
            st.error("CSV must contain a column named `tweet`.")
        else:
            st.write(f"Loaded **{len(df_up)}** tweets. Preview:")
            st.dataframe(df_up.head())

            if st.button("Analyze All", type="primary", key="batch"):
                results  = []
                progress = st.progress(0)
                status   = st.empty()

                for i, row in df_up.iterrows():
                    result = predict(str(row["tweet"]))
                    results.append({
                        "tweet"            : row["tweet"],
                        "topic"            : result["topic"],
                        "predicted_emotion": f"{result['emoji']} {result['emotion']}",
                        "confidence (%)"   : round(result["confidence"], 2)
                    })
                    progress.progress((i + 1) / len(df_up))
                    status.text(f"Processing {i + 1} / {len(df_up)}...")

                progress.empty()
                status.empty()

                df_results = pd.DataFrame(results)
                st.success(f"Done! Analyzed **{len(df_results)}** tweets.")
                st.dataframe(df_results)

                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=df_results.to_csv(index=False).encode("utf-8"),
                    file_name="emotion_results.csv",
                    mime="text/csv"
                )