import joblib
import argparse
import pandas as pd
from data_processing import load_and_preprocess

def run_fairness_audit(check="sex", model_type="random_forest"):
    model = joblib.load(f"results/{model_type}_model.pkl")
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()

    predictions = model.predict(X_test)

    audit_df = demo_test.copy()
    audit_df["prediction"] = predictions

    if check == "age":
        audit_df["group"] = audit_df["Age"].apply(lambda x: "under_30" if x < 30 else "30_and_above")
    elif check == "sex":
        audit_df["group"] = audit_df["Sex"]
    else:
        raise ValueError("check must be 'age' or 'sex'")

    approval_rates = audit_df.groupby("group")["prediction"].mean()

    print(f"\nFairness Audit — grouped by: {check}")
    print("-" * 40)
    for group, rate in approval_rates.items():
        print(f"  {group}: {rate*100:.1f}% approval rate")

    gap = approval_rates.max() - approval_rates.min()
    print(f"\nApproval rate gap: {gap*100:.1f}%")

    if gap > 0.10:
        verdict = "WARNING: Potential fairness concern - gap exceeds 10%"
    else:
        verdict = "OK: No major fairness concern detected (gap under 10%)"
    print(verdict)

    with open("results/audit_report.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Fairness Audit - {check}\n")
        for group, rate in approval_rates.items():
            f.write(f"- {group}: {rate*100:.1f}% approval rate\n")
        f.write(f"- Approval rate gap: {gap*100:.1f}%\n")
        f.write(f"- Verdict: {verdict}\n")

    print("\nResults appended to results/audit_report.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", type=str, default="sex", choices=["age", "sex"])
    args = parser.parse_args()

    run_fairness_audit(check=args.check)