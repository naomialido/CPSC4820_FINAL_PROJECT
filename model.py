import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import joblib

# load data
data = pd.read_csv("data/Telco_customer_churn.csv")


# drop 'Churn Reason' if it exists
data = data.drop(["Churn Reason"], axis=1, errors="ignore")

# convert numeric column
data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")

# drop missing values
data = data.dropna()


# engineer Location Type from Zip Code
zip_density = pd.read_csv("data/uszips.csv", usecols=["zip", "density"])
zip_density["zip"] = zip_density["zip"].astype(str).str.zfill(5)
data["Zip Code"] = data["Zip Code"].astype(str).str.zfill(5)

data = data.merge(zip_density, left_on="Zip Code", right_on="zip", how="left")


def classify_density(density):
    if pd.isna(density):
        return "Suburban"
    if density >= 2500:
        return "Urban"
    elif density >= 1000:
        return "Suburban"
    else:
        return "Rural"


data["Location Type"] = data["density"].apply(classify_density)
data = data.drop(columns=["zip", "density"])


print("Engineering Location Type from Zip Code...")
print(data["Location Type"].value_counts())

# feature engineering
categorical_cols = [
    "Location Type",
    "Senior Citizen",
    "Gender",
    "Partner",
    "Dependents",
    "Phone Service",
    "Multiple Lines",
    "Internet Service",
    "Online Security",
    "Online Backup",
    "Device Protection",
    "Tech Support",
    "Streaming TV",
    "Streaming Movies",
    "Contract",
    "Paperless Billing",
    "Payment Method",
]

data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# drop non-predictive columns before encoding
data = data.drop(
    [
        "CustomerID",
        "Count",
        "Lat Long",
        "Latitude",
        "Longitude",
        "Zip Code",
        "City",
        "State",
        "Country",
        "CLTV",
        "Churn Label",
        "Churn Score",
        "Churn Reason",
    ],
    axis=1,
    errors="ignore",
)

# Features and target
X = data.drop(["Churn Value"], axis=1, errors="ignore")
y = data["Churn Value"]


# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# handle imbalance with SMOTE
smote = SMOTE(sampling_strategy="minority", random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# train models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)

lr.fit(X_train_sm, y_train_sm)
rf.fit(X_train_sm, y_train_sm)


# evaluation
print("\nLogistic Regression Report")
print(classification_report(y_test, lr.predict(X_test)))

print("\nRandom Forest Report")
print(classification_report(y_test, rf.predict(X_test)))


# save best model (random forest)
joblib.dump(rf, "churn_model.pkl")

# save feature columns
joblib.dump(X.columns.tolist(), "model_columns.pkl")

print("Model and columns saved!")


# generate predictions for tableau
# Ensure full dataset has all columns
for col in X.columns:
    if col not in data.columns:
        data[col] = 0

# Select model features
data_for_pred = data[X.columns]

# Predict churn and probability
data["Churn_Pred"] = rf.predict(data_for_pred)
data["Churn_Prob"] = rf.predict_proba(data_for_pred)[:, 1]

# Save to CSV
data.to_csv("data/predictions_for_tableau.csv", index=False)
print("CSV with predictions saved as predictions_for_tableau.csv")
