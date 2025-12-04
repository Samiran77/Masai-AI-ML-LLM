import sys
from pathlib import Path

import pandas as pd

TRAIN_PATH = Path('./data/processed/train.csv')

EXPECTED_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
    "target"
]

def main() -> int:
    if not TRAIN_PATH.exists():
        print(f"Error: {TRAIN_PATH} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    df = pd.read_csv(TRAIN_PATH)
    
    #Column check
    cols = list[any](df.columns)
    if(cols != EXPECTED_COLUMNS):
        print(f"Error: {TRAIN_PATH} does not contains the expected columns.")
        return 1;

    #NULL check
    null_counts = df.isnull().sum()
    if(null_counts.any()):
        print(f"Error: {TRAIN_PATH} contains null values:\n")
        return 1;
    
    print(f"[OK] Data validation passed for {TRAIN_PATH}")
    
    return 0;

if __name__ == "__main__":
    sys.exit(main())
    