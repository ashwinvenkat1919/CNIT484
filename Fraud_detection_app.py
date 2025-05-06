import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import plotly.express as px

# Constants
MODEL_DIR = "bert_source_code/distilbert_fraud_detection"
MAX_LENGTH = 128
DEVICE = torch.device("cpu")

# Load model
@st.cache_resource
def load_model():
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(DEVICE)
    model.eval()
    return model

# Load tokenizer
@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(MODEL_DIR)

# Load and merge data
@st.cache_data
def load_data():
    transactions = pd.read_csv("bert_source_code/distilbert_fraud_detection/archive/transactions_data.csv")
    cards = pd.read_csv("bert_source_code/distilbert_fraud_detection/archive/cards_data.csv").rename(columns={"id": "card_id"})
    users = pd.read_csv("bert_source_code/distilbert_fraud_detection/archive/users_data.csv").rename(columns={"id": "user_id"})

    # Merge transactions -> users first
    merged = transactions.merge(users, left_on="client_id", right_on="user_id", how="left")

    # Then merge with cards
    merged = merged.merge(cards, on="card_id", how="left")

    # Drop rows missing model-required fields
    merged.dropna(subset=["merchant_city", "merchant_state", "zip", "mcc", "errors"], inplace=True)

    # Create input text
    merged["text"] = merged.apply(
        lambda row: f"{row['merchant_city']} {row['merchant_state']} {row['zip']} MCC:{row['mcc']} Errors:{row['errors']}",
        axis=1
    )

    return merged

# Batched prediction for efficiency

def predict_batch(texts, model, tokenizer):
    # Tokenize all texts in one batch
    encodings = tokenizer.batch_encode_plus(texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH)
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}

    with torch.no_grad():
        logits = model(**encodings).logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_labels = torch.argmax(probs, dim=1).tolist()
        pred_confidences = probs.max(dim=1).values.tolist()

    return pred_labels, [round(conf, 4) for conf in pred_confidences]

# Main app

def main():
    st.title("\U0001F4B3 Fraud Detection Dashboard with DistilBERT")

    model = load_model()
    tokenizer = load_tokenizer()
    df = load_data()

    st.write("### \U0001F9FE Preview of Transactions Dataset")
    st.dataframe(df.head())

    st.write("### \U0001F50D Predict a Sample Transaction")
    sample_text = st.selectbox("Select a transaction text", df["text"].sample(10).tolist())
    if st.button("Predict Fraud"):
        label, confidence = predict_batch([sample_text], model, tokenizer)
        label_str = '🛑 Fraudulent' if label[0] == 1 else '✅ Legitimate'
        st.success(f"Prediction: {label_str} (Confidence: {confidence[0] * 100:.2f}%)")

    st.write("### \U0001F9E0 Run Balanced Sample Predictions (100 Rows)")
    if st.button("Run Predictions on Balanced Sample"):
        with st.spinner("Running predictions on a sample of 100 (50 fraud + 50 legit)..."):
            df[["prediction", "confidence"]] = df["text"].apply(lambda x: pd.Series(predict_batch([x], model, tokenizer))).explode().unstack().astype({'prediction': 'int', 'confidence': 'float'})
            fraud = df[df["prediction"] == 1].sample(min(50, len(df[df["prediction"] == 1])), random_state=1)
            legit = df[df["prediction"] == 0].sample(min(50, len(df[df["prediction"] == 0])), random_state=1)
            sample_df = pd.concat([fraud, legit]).copy()

        st.success("✅ Predictions complete!")

        st.write("### 1️⃣ Fraud Prediction Distribution")
        st.bar_chart(sample_df["prediction"].value_counts())

        st.write("### 2️⃣ Top Cities with Fraud")
        top_cities = sample_df[sample_df["prediction"] == 1]["merchant_city"].value_counts().head(10)
        st.plotly_chart(px.bar(top_cities, title="Top Fraud-Prone Cities", labels={"value": "Fraud Count", "index": "City"}))

        st.write("### 3️⃣ Zip Codes with Highest Fraud Rates")
        zip_pct = sample_df.groupby("zip")["prediction"].mean().sort_values(ascending=False).head(10) * 100
        st.plotly_chart(px.bar(zip_pct, title="Top Zip Codes by % Fraud", labels={"value": "% Fraud", "index": "Zip Code"}))

        st.write("### 4️⃣ Fraud by Card Type")
        cardtype_chart = sample_df[sample_df["prediction"] == 1]["card_type"].value_counts()
        st.plotly_chart(px.bar(cardtype_chart, title="Fraud Count by Card Type", labels={"value": "Fraud Count", "index": "Card Type"}))

        st.write("### 5️⃣ Fraud by Gender")
        gender_chart = sample_df[sample_df["prediction"] == 1]["gender"].value_counts()
        st.plotly_chart(px.bar(gender_chart, title="Fraud Count by Gender", labels={"value": "Fraud Count", "index": "Gender"}))

        st.write("### 6️⃣ Fraud by Age Group")
        sample_df["age_group"] = pd.cut(sample_df["current_age"], bins=[0, 25, 40, 55, 70, 100], labels=["<25", "25-40", "40-55", "55-70", "70+"])
        age_chart = sample_df[sample_df["prediction"] == 1]["age_group"].value_counts().sort_index()
        st.plotly_chart(px.bar(age_chart, title="Fraud Count by Age Group", labels={"value": "Fraud Count", "index": "Age Group"}))

        sample_df["word_count"] = sample_df["text"].apply(lambda x: len(str(x).split()))
        st.write("### 7️⃣ Word Count vs Fraud Prediction")
        st.plotly_chart(px.box(sample_df, x="prediction", y="word_count", color="prediction",
                               title="Text Word Count Distribution by Prediction",
                               labels={"prediction": "Prediction (0=Legit, 1=Fraud)", "word_count": "Word Count"}))

        st.write("### 8️⃣ Confidence Score Distribution")
        st.plotly_chart(px.histogram(sample_df, x="confidence", color="prediction",
                                     nbins=20, title="Model Confidence Distribution",
                                     labels={"confidence": "Confidence", "prediction": "Prediction"}))

if __name__ == "__main__":
    main()
