import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import random
from config import DATA_PATH

def load_and_process_data():
    # 加载数据
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        # 模拟数据
        data = {
            "id": range(1000),
            "date": ["2016-06-11 06:29:00"] * 1000,
            "client_id": [random.randint(1000, 2000) for _ in range(1000)],
            "card_id": [random.randint(2000, 3000) for _ in range(1000)],
            "amount": [round(random.uniform(-500, 500), 2) for _ in range(1000)],
            "use_chip": ["Chip Transaction"] * 1000,
            "merchant_id": [random.randint(80000, 90000) for _ in range(1000)],
            "merchant_city": ["San Antonio"] * 1000,
            "merchant_state": ["PA"] * 1000,
            "zip": [38394] * 1000,
            "mcc": [8043] * 1000,
            "mcc_description": ["Optometrists, Optical Goods and Eyeglasses"] * 1000,
            "errors": ["DECLINE CODE 203"] * 1000,
            "is_fraud": [random.choice([0, 1]) for _ in range(1000)]
        }
        df = pd.DataFrame(data)

    # 创建提示
    def create_prompt(row):
        return f"""Transaction details:
- Amount: {row['amount']}
- Use Chip: {row['use_chip']}
- Merchant City: {row['merchant_city']}
- Merchant State: {row['merchant_state']}
- ZIP: {row['zip']}
- MCC Description: {row['mcc_description']}
- Errors: {row['errors']}
Is this transaction fraudulent? Respond with 'True' or 'False'."""

    df['prompt'] = df.apply(create_prompt, axis=1)
    df['label'] = df['is_fraud'].apply(lambda x: "True" if x == 1 else "False")

    # 分割数据集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = Dataset.from_pandas(train_df[['prompt', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['prompt', 'label']])

    return train_dataset, test_dataset, test_df