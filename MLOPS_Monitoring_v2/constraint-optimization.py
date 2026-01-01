from fairlearn.reductions import ConstraintOptimization, DemographicParity
from sklearn.linear_model import LogisticRegression

base_model = LogisticRegression(solver='liblinear', itermax=1000)

constraint = DemographicParity()

mitigator = ConstraintOptimization(
    estimator=base_model,
    constraints=constraint,
#    grid_size=100,
#    eps=0.01
)

mitigator.fit(X_train, y_train, sensitive_features=X_train['gender'])

y_pred = mitigator.predict(X_test)

print("Predictions after applying Constraint Optimization for Demographic Parity:" + str(y_pred))