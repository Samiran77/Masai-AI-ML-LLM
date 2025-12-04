import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

RAW_DATA_PATH = Path('./data/raw/iris.csv')
PROCESSED_DIR = Path('./data/processed/')

def prepare_data():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_DATA_PATH)
    
    train_df, test_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=42,
        stratify=df['target']
        )
    
    train_df.to_csv(PROCESSED_DIR / 'train.csv', index=False)
    test_df.to_csv(PROCESSED_DIR / 'test.csv', index=False)

if __name__ == "__main__":
    prepare_data()
    