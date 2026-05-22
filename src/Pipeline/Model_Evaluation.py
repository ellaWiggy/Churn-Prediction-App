# Usual imports
import os
import sys
import pandas as pd

# Custom logger and exception imports
from src.utils.file_logs import get_logger
from src.utils.exceptions import CustomException

# Sklearn imports for evaluation
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

# Plots
import matplotlib.pyplot as plt
import seaborn as sns
import shap

#Custome imports for evaluation
from src.utils.Modeling import evaluate_classifier

log = get_logger()

def evaluate_model(model, test_data):
    try:
        log.info("Starting model evaluation.")
        # load the test data and separate features and target variable
        X_test = test_data.drop(columns=['churn'], axis=1)
        y_test = test_data['churn']
        
        # Make predictions and get predicted probabilities
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        # Evaluate using the custom evaluate_classifier function
        results = evaluate_classifier(y_test, y_pred, y_proba)
        log.info("\n ---------- CLASSIFIER EVALUATION REPORT ----------\n")
        log.info('\n' + classification_report(y_test, y_pred))
        log.info(f"AUC-ROC: {results['auc_roc']:.4f}")
        log.info(f"\nConfusion Matrix:\n  TN: {results['tn']:6d} | FP: {results['fp']:6d}\n  FN: {results['fn']:6d} | TP: {results['tp']:6d}")

        # Confusion Matrix Plot
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig("artifacts/confusion_matrix.png")
        plt.close()

        # SHAP Summary Plot
        # Feature names for SHAP
        transformer_names = model.named_steps['transformation'].get_feature_names_out()
        selected_mask = model.named_steps['selection'].support_
        # Match SHAP labels to selected features so plots stay correct.
        final_feature_names = [name for name, selected in zip(transformer_names, selected_mask) if selected]

        # Set up test data for SHAP
        X_test_transformed = model[:-1].transform(X_test)
        X_test_df = pd.DataFrame(X_test_transformed, columns=final_feature_names)
        fitted_lgbm = model.named_steps['model']

        # Calculate SHAP values
        explainer = shap.TreeExplainer(fitted_lgbm)
        shap_values = explainer.shap_values(X_test_df)

        # create SHAP summary plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        plt.sca(ax1) 
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False, plot_size=None)
        ax1.set_title("Global Feature Importance", fontsize=16)
        ax1.set_xlabel("mean(|SHAP value|)\n(avg impact on output)", fontsize=10)
        ax1.tick_params(axis='x', labelsize=9)
        plt.sca(ax2) 
        shap.summary_plot(shap_values, X_test_df, show=False, plot_size=None)
        ax2.set_title("Feature Impact (Directionality)", fontsize=16)
        ax2.set_yticklabels([]) 
        ax2.set_ylabel("") 
        ax2.set_xlabel("SHAP value\n(impact on model output)", fontsize=10)
        ax2.tick_params(axis='x', labelsize=9)
        plt.subplots_adjust(bottom=0.2, wspace=0.1)
        plt.savefig("artifacts/shap_summary.png")
        plt.close()

        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred), 
            "f1": f1_score(y_test, y_pred),
            "auc_roc": results['auc_roc']
        }

    except Exception as e:
        log.error(f"Error during model evaluation: {e}")
        raise CustomException(e, sys)