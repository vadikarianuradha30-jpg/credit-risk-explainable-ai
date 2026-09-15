import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath="data/german_credit_data.csv"):
    # Load data
    df = pd.read_csv(filepath)

    # Drop the leftover index column
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    # Handle missing values (Saving accounts / Checking account often have NA)
    df["Saving accounts"] = df["Saving accounts"].fillna("unknown")
    df["Checking account"] = df["Checking account"].fillna("unknown")

    # Keep demographic columns aside for fairness audit later
    demo = df[["Age", "Sex"]].copy()

    # Encode target: good=1 (approve), bad=0 (reject)
    df["Risk"] = df["Risk"].map({"good": 1, "bad": 0})

    # Separate features and target
    X = df.drop(columns=["Risk"])
    y = df["Risk"]

    # Encode categorical columns
    categorical_cols = X.select_dtypes(include="object").columns
    for col in categorical_cols:
        X[col] = LabelEncoder().fit_transform(X[col])

    # Split into train/test, keeping matching demo split
    X_train, X_test, y_train, y_test, demo_train, demo_test = train_test_split(
        X, y, demo, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, demo_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()
    print("Train shape:", X_train.shape)
    print("Test shape:", X_test.shape)
    print("Sample features:\n", X_train.head())
    print("Sample target:\n", y_train.head())