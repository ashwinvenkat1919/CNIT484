# 配置参数
MODEL_NAME = "distilbert-base-uncased"  # 切换到 DistilBERT
DATA_PATH = "dataset.csv"
OUTPUT_DIR = "./distilbert_fraud_detection"
MAX_LENGTH = 128  # 减小，分类任务提示较短
BATCH_SIZE = 8
NUM_EPOCHS = 3