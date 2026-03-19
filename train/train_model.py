import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# Load dataset
df = pd.read_csv("data/credit_data.csv")

df.columns = [
    "checking_account",
    "duration",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings",
    "employment",
    "installment_rate",
    "personal_status",
    "other_debtors",
    "residence",
    "property",
    "age",
    "other_installment",
    "housing",
    "existing_credits",
    "job",
    "maintenance_people",
    "telephone",
    "foreign_worker",
    "default"
]

df["default"] = df["default"].map({1: 0, 2: 1})

X = df.drop("default", axis=1)
y = df["default"]

X = pd.get_dummies(X, drop_first=True)


# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Train model
model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# jumlah feature
n_features = X_train.shape[1]

initial_type = [("input", FloatTensorType([None, n_features]))]

onnx_model = convert_sklearn(model, initial_types=initial_type)


# Evaluation
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\nModel Evaluation")
print("====================")

print(classification_report(y_test, y_pred))

auc = roc_auc_score(y_test, y_prob)

print("ROC AUC:", round(auc, 4))


# Save model
joblib.dump(model, "model/credit_model.pkl")


# Save feature name for inference
joblib.dump(X.columns.tolist(), "model/features.pkl")


# Save ONNX model
with open("model/credit_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())


print("\nModel saved to: model/credit_model.pkl")
print("Feature list saved to: model/features.pkl")
print("ONNX model saved to: model/credit_model.onnx")