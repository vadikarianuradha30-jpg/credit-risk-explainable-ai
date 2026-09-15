import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import argparse
from data_processing import load_and_preprocess

def get_shap_values_for_class(shap_values, class_index=1):
    if isinstance(shap_values, list):
        return shap_values[class_index]
    elif shap_values.ndim == 3:
        return shap_values[:, :, class_index]
    else:
        return shap_values


def explain_predictions(applicant_id=None, summary=False, model_type="random_forest"):
    model = joblib.load(f"results/{model_type}_model.pkl")
    X_train, X_test, y_train, y_test, demo_test = load_and_preprocess()

    explainer = shap.TreeExplainer(model)
    raw_shap_values = explainer.shap_values(X_test)
    shap_values_to_use = get_shap_values_for_class(raw_shap_values, class_index=1)

    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[1]
    else:
        base_value = explainer.expected_value

    if summary:
        plt.figure()
        shap.summary_plot(shap_values_to_use, X_test, show=False)
        plt.tight_layout()
        plt.savefig("results/shap_summary_plot.png")
        print("Summary plot saved to results/shap_summary_plot.png")

    if applicant_id is not None:
        idx = applicant_id
        if idx >= len(X_test):
            print(f"Applicant id {idx} out of range (test set has {len(X_test)} rows). Using 0 instead.")
            idx = 0

        prediction = model.predict(X_test.iloc[[idx]])[0]
        label = "APPROVED" if prediction == 1 else "REJECTED"

        print(f"\nApplicant {idx}: {label}")
        print("Top contributing factors:")

        applicant_shap = shap_values_to_use[idx]
        feature_impact = list(zip(X_test.columns, applicant_shap))
        feature_impact.sort(key=lambda pair: abs(pair[1]), reverse=True)

        for feature, impact in feature_impact[:5]:
            direction = "+" if impact > 0 else "-"
            print(f"  {feature}: {direction}{abs(impact):.3f}")

        plt.figure()
        shap.plots._waterfall.waterfall_legacy(
            base_value,
            applicant_shap,
            feature_names=X_test.columns,
            show=False
        )
        plt.tight_layout()
        plt.savefig(f"results/shap_applicant_{idx}.png")
        print(f"Waterfall plot saved to results/shap_applicant_{idx}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--applicant_id", type=int, default=None)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()

    explain_predictions(applicant_id=args.applicant_id, summary=args.summary)