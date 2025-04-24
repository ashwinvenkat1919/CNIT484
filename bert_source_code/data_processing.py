import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
import random
from config import DATA_PATH

def load_and_process_data():
    try:
        df = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print("CSV not found, generating sample data...")
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

    # 数据清理
    df['amount'] = df['amount'].replace(r'[\$]', '', regex=True).astype(float)
    df['errors'] = df['errors'].fillna("None")
    df['merchant_city'] = df['merchant_city'].fillna("Unknown")
    df['merchant_state'] = df['merchant_state'].fillna("Unknown")
    df['mcc_description'] = df['mcc_description'].fillna("Unknown")
    df['use_chip'] = df['use_chip'].fillna("Unknown")
    df['zip'] = df['zip'].fillna(0).astype(int)
    df = df[df['is_fraud'].isin([0, 1])]

    # 打印数据分布
    print("is_fraud distribution:\n", df['is_fraud'].value_counts())
    print("Geographic inconsistencies:\n", df[['merchant_city', 'merchant_state']].drop_duplicates())

    def create_prompt(row):
        return f"""Amount: {row['amount']:.2f}, Use Chip: {row['use_chip']}, Merchant City: {row['merchant_city']}, Merchant State: {row['merchant_state']}, ZIP: {row['zip']}, MCC Description: {row['mcc_description']}, Errors: {row['errors']}"""

    df['text'] = df.apply(create_prompt, axis=1)
    df['label'] = df['is_fraud'].astype(int)  # 分类模型需要 0/1 标签

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['is_fraud'])
    train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
    test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

    return train_dataset, test_dataset, test_df