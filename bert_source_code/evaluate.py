#!/usr/bin/env python
# bert_source_code/evaluate.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from config import OUTPUT_DIR, MAX_LENGTH

def evaluate_model(test_df):
    model = AutoModelForSequenceClassification.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def predict_fraud(text):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH
        ).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        pred = torch.argmax(logits, dim=1).item()
        return "True" if pred == 1 else "False"

    y_pred = [predict_fraud(t) for t in test_df['text']]
    y_true = test_df['label'].map(lambda x: "True" if x == 1 else "False").tolist()

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', pos_label="True"
    )

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # 打印一个样例
    sample = test_df.iloc[0]
    print("\n--- Sample ---")
    print(sample['text'])
    print(f"True Label: {'True' if sample['label']==1 else 'False'}")
    print(f"Predicted : {predict_fraud(sample['text'])}")

if __name__ == "__main__":
    # 如果直接运行此脚本，就去加载数据并评估
    from data_processing import load_and_process_data

    print("► Loading test data...")
    _, _, test_df = load_and_process_data()
    print("► Running evaluation...")
    evaluate_model(test_df)
