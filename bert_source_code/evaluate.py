#!/usr/bin/env python
# bert_source_code/evaluate.py

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch
from config import OUTPUT_DIR, MAX_LENGTH, MODEL_NAME

def evaluate_model(test_df, model_path=OUTPUT_DIR, is_pretrained=False):
    """评估模型性能
    
    Args:
        test_df: 测试数据集
        model_path: 模型路径，可以是微调后的模型或原始模型
        is_pretrained: 是否为原始预训练模型(未微调)
    """
    # 加载模型和tokenizer
    if is_pretrained:
        # 对于原始预训练模型，我们需要指定num_labels
        model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=2)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
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

    model_type = "Pretrained (no fine-tuning)" if is_pretrained else "Fine-tuned"
    print(f"\n=== {model_type} Model Evaluation ===")
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")

    # 打印一个样例
    sample = test_df.iloc[0]
    print("\n--- Sample ---")
    print(sample['text'])
    print(f"True Label: {'True' if sample['label']==1 else 'False'}")
    print(f"Predicted : {predict_fraud(sample['text'])}")

def evaluate_both_models(test_df):
    """评估微调前后的模型性能"""
    print("\nEvaluating fine-tuned model...")
    evaluate_model(test_df, OUTPUT_DIR, is_pretrained=False)
    
    print("\nEvaluating original pretrained model (zero shot)...")
    evaluate_model(test_df, MODEL_NAME, is_pretrained=True)

if __name__ == "__main__":
    # 如果直接运行此脚本，就去加载数据并评估
    from data_processing import load_and_process_data

    print("► Loading test data...")
    _, _, test_df = load_and_process_data()
    print("► Running evaluation...")
    evaluate_both_models(test_df)