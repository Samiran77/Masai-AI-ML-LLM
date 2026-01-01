import numpy as np

train_df = X_train.copy()
train_df['target'] = y_train.values
train_df['gender'] = X_train['gender'].values

group_positive_rates = train_df.groupby('gender')['target'].mean()
overall_positive_rate = train_df['target'].mean()

weights = {}

for _, row in train_df.iterrows():
    g=row['gender']
    w = (overall_positive_rate / group_positive_rates[g]) 
    weights.append(w)

sample_weights = np.array(weights)

print("Sample weights calculated for reweighing:" + str(sample_weights))