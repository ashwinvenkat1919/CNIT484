from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import accuracy_score
import torch
from config import OUTPUT_DIR, MAX_LENGTH

def evaluate_model(test_df):
    # 加载微调模型
    model = AutoModelForCausalLM.from_pretrained(OUTPUT_DIR)
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 预测函数
    def predict_fraud(prompt):
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LENGTH).to(device)
        outputs = model.generate(**inputs, max_new_tokens=1)
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # 测试集预测
    predictions = [predict_fraud(prompt) for prompt in test_df['prompt']]
    true_labels = test_df['label'].tolist()
    accuracy = accuracy_score(true_labels, predictions)

    # 输出结果
    print(f"Test Accuracy: {accuracy:.4f}")
    sample = test_df.iloc[0]
    print("\nSample Transaction:")
    print(sample['prompt'])
    print(f"True Label: {sample['label']}")
    print(f"Predicted: {predict_fraud(sample['prompt'])}")