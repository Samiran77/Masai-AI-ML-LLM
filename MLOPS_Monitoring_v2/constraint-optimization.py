import pandas as pd
from fairlearn.reductions import ConstraintOptimization, DemographicParity
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from sklearn.model_selection import train_test_split

RAW_DATA_PATH = Path('./data/raw/iris.csv')
PROCESSED_DIR = Path('./data/processed/')


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


TRAIN_PATH = Path('./data/processed/train.csv')
TEST_PATH = Path('./data/processed/test.csv')



base_model = LogisticRegression(solver='liblinear', itermax=1000)

constraint = DemographicParity()

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

X_train = train_df.drop(columns=['target', 'petal length (cm)'])
y_train = train_df['target']
X_test = test_df.drop(columns=['target', 'petal length (cm)'])
y_test = test_df['target']
sensitive_feature = train_df['petal length (cm)']

mitigator = ConstraintOptimization(
    estimator=base_model,
    constraints=constraint,
#    grid_size=100,
#    eps=0.01
)

#mitigator.fit(X_train, y_train, sensitive_features=X_train['gender'])
mitigator.fit(X_train, y_train, sensitive_features=sensitive_feature)

y_pred = mitigator.predict(X_test)

print("Predictions after applying Constraint Optimization for Demographic Parity:" + str(y_pred))