import pandas as pd
from sklearn.datasets import load_iris

def main():
    iris = load_iris(as_frame=True)
    df = iris.frame
    df.to_csv('./data/raw/iris.csv', index=False)
    
if __name__ == "__main__":
    main()