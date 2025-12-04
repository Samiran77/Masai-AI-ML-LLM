import json
from pathlib import Path
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

TRAIN_PATH = Path('./data/processed/train.csv')
TEST_PATH = Path('./data/processed/test.csv')
MODEL_DIR = Path('./models/')
METRICS_DIR = Path('./metrics/')

def get_git_commit_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('utf-8')
    except Exception:   
        return "unknown"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    
    X_train = train_df.drop(columns=['target'])
    y_train = train_df['target']
    X_test = test_df.drop(columns=['target'])
    y_test = test_df['target']
    
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    
    model_path = MODEL_DIR / 'iris_logistics_regression.pkl'
    joblib.dump(model, model_path)
    
    metrics = {
        'accuracy': accuracy,
        'git_commit_hash': get_git_commit_hash()
    }
    
    metrics_path = METRICS_DIR / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f)
    
    print(f"[OK] Model trained and metrics saved to {model_path}")
    print(f"[OK] Accuracy Score: {accuracy}")
    print(f"[OK] Git Commit Hash: {get_git_commit_hash()}")
    
if __name__ == "__main__":
    main()