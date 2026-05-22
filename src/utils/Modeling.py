# Data manipulation and analysis
import pandas as pd
import numpy as np

# Modelling.
from sklearn.model_selection import cross_val_score, StratifiedKFold, cross_validate
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import(roc_auc_score, 
                            classification_report, 
                            confusion_matrix,
                            accuracy_score,
                            precision_score,
                            recall_score,
                            f1_score)

# Hyperparameter optimization
import optuna

# Suppress warnings and logging for cleaner output
import logging
import warnings
warnings.filterwarnings('ignore')
logging.getLogger('lightgbm').setLevel(logging.ERROR)
logging.getLogger('sklearn').setLevel(logging.ERROR)

    
class FeatureEngineering(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_engineer=None):
        self.columns_to_engineer = columns_to_engineer
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        temp_df = self._transform_math_logic(X.head(1))
        self.feature_names_out_ = temp_df.columns.tolist()
        return self

    def transform(self, X):
        return self._transform_math_logic(X)
        
    def _transform_math_logic(self, X):
        X_copy = X.copy()

        discrete = ['dependent_count', 'months_on_book', 'total_relationship_count', 
                    'months_inactive_12_mon', 'contacts_count_12_mon', 'total_trans_ct']
        continuous = ['credit_limit', 'total_revolving_bal', 'total_amt_chng_q4_q1', 
                      'total_trans_amt', 'total_ct_chng_q4_q1', 'avg_utilization_ratio']
        
        existing_discrete = [col for col in discrete if col in X_copy.columns]
        existing_continuous = [col for col in continuous if col in X_copy.columns]

        X_copy[existing_discrete] = X_copy[existing_discrete].astype('int32')
        X_copy[existing_continuous] = X_copy[existing_continuous].astype('float32')

        eps = 1e-9
        if 'total_revolving_bal' in X_copy.columns and 'credit_limit' in X_copy.columns:
            X_copy['credit_util_rate'] = X_copy['total_revolving_bal'] / (X_copy['credit_limit'] + eps)
        if 'months_inactive_12_mon' in X_copy.columns and 'months_on_book' in X_copy.columns:
            X_copy['prop_inactive_months'] = X_copy['months_inactive_12_mon'] / (X_copy['months_on_book'] + eps)
        if 'total_trans_amt' in X_copy.columns and 'total_trans_ct' in X_copy.columns:
            X_copy['avg_trans_amt'] = X_copy['total_trans_amt'] / (X_copy['total_trans_ct'] + eps)
        if 'total_trans_amt' in X_copy.columns and 'months_on_book' in X_copy.columns:
            X_copy['trans_amt_per_months'] = X_copy['total_trans_amt'] / (X_copy['months_on_book'] + eps)
        if 'total_trans_ct' in X_copy.columns and 'months_on_book' in X_copy.columns:
            X_copy['trans_ct_velocity'] = X_copy['total_trans_ct'] / (X_copy['months_on_book'] + eps)
        if 'avg_utilization_ratio' in X_copy.columns and 'total_amt_chng_q4_q1' in X_copy.columns:
            X_copy['util_chng_use'] = X_copy['avg_utilization_ratio'] * X_copy['total_amt_chng_q4_q1']
        
        if 'months_inactive_12_mon' in X_copy.columns:
            X_copy['high_inactive'] = (X_copy['months_inactive_12_mon'] >= 4).astype('int8')
        if 'contacts_count_12_mon' in X_copy.columns:
            X_copy['low_contact_freq'] = (X_copy['contacts_count_12_mon'] <= 2).astype('int8')
        if 'total_amt_chng_q4_q1' in X_copy.columns:
            X_copy['bal_decrease'] = (X_copy['total_amt_chng_q4_q1'] < 0).astype('int8')
            
        if 'high_inactive' in X_copy.columns and 'low_contact_freq' in X_copy.columns:
            X_copy['churn_warning_score']   = (X_copy['high_inactive'] + X_copy['low_contact_freq']).astype('int8')
        

        edu_map = {'Uneducated': 1, 'High School': 2, 'College': 3, 'Graduate': 4, 'Post-Graduate': 5, 'Doctorate': 6, 'Unknown': 0}
        inc_map = {'Less than $40K': 1, '$40K - $60K': 2, '$60K - $80K': 3, '$80K - $120K': 4, '$120K +': 5, 'Unknown': 0}
        
        edu_val = X_copy['education_level'].map(edu_map).fillna(0)
        inc_val = X_copy['income_category'].map(inc_map).fillna(0)
        X_copy['education_income_index'] = (edu_val + inc_val).astype('int32')

        return X_copy

    def get_feature_names_out(self, input_features=None):
        if self.feature_names_out_ is None:
            return np.array(input_features, dtype=object)
        return np.array(self.feature_names_out_, dtype=object)

def evaluate_models(models, X, y, n_folds=5):
# Evaluates multiple models against each other
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    results = []

    for name, model in models.items():
        print(f"Training and evaluating: {name}...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_res = cross_validate(model, X, y, cv=skf, scoring=metrics, n_jobs=-1)

        res_dict = {
            'Model Name': name,
            'Accuracy': cv_res['test_accuracy'].mean(),
            'Precision': cv_res['test_precision'].mean(),
            'Recall': cv_res['test_recall'].mean(),
            'F1 Score': cv_res['test_f1'].mean(),
            'AUC-ROC': cv_res['test_roc_auc'].mean(),
            'Time (s)': cv_res['fit_time'].mean()
        }
        results.append(res_dict)

    comparison_df = pd.DataFrame(results).sort_values(by='AUC-ROC', ascending=False)
    print(f"{'\n------- Model Comparison -------'}")
 
    
    print(comparison_df.to_string(index=False, justify='center', float_format=lambda x: f"{x:.4f}"))

    print("-" * 14)
    best_model = comparison_df.iloc[0]['Model Name']
    best_auc = comparison_df.iloc[0]['AUC-ROC']
    print(f"Best Model: {best_model} with an AUC-ROC of {best_auc:.4f}")
    print("-" * 14)

    return comparison_df

def evaluate_classifier(y_true, y_pred, y_pred_proba, target_names=['Non-Churner', 'Churner']):
    # Evaluate Classifier performance
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    results = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'tn': tn, 
        'fp': fp, 
        'fn': fn, 
        'tp': tp
    }

    print(" ------------- Classifier Evaluation -------------")
    print(classification_report(y_true, y_pred, target_names=target_names))
    print(f"Final AUC-ROC: {results['auc_roc']:.4f}")
    
    print(f"\nConfusion Matrix Breakdown:")
    print(f"True Neg: {tn} | False Pos: {fp}")
    print(f"False Neg: {fn} | True Pos: {tp}")
    
    return results

def feature_selector(estimator, step=3, folds=2, min_features_to_select=1):

    selector = RFECV(
        estimator=estimator,
        step=step,
        cv=StratifiedKFold(folds, shuffle=True, random_state=42),
        scoring='f1',
        min_features_to_select=min_features_to_select,
        n_jobs=-1
    )
    selector.set_output(transform="pandas")
    return selector

def hyperparameter_tuning(X, y, pipeline, n_trials=50, cv=5, scoring='f1'):
    # Hyperparameter tuning using Optuna for a given model class, optimizing for a specified scoring metric
    def objective(trial):
        min_feats = trial.suggest_int('min_features_to_select', 10, 30)

        params = {
            'model__learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
            'model__n_estimators': trial.suggest_int('n_estimators', 500, 1000, step=100),
            'model__num_leaves': trial.suggest_int('num_leaves', 20, 100),
            'model__max_depth': trial.suggest_int('max_depth', 3, 12),
            'model__min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
            'model__reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'model__reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'model__colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 0.9),
            'model__scale_pos_weight': trial.suggest_float('scale_pos_weight', 1.0, 6.0)
        }

        pipeline.set_params(**params, selection__min_features_to_select=min_feats)

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        scores = []
        for i, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipeline.fit(X_train, y_train)
            y_preds = pipeline.predict(X_val)
            score = f1_score(y_val, y_preds)
            scores.append(score) 

            trial.report(score, i)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        return sum(scores) / len(scores)

    study = optuna.create_study(direction="maximize", pruner = optuna.pruners.MedianPruner(n_startup_trials=5))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\nBest {scoring.upper()}: {study.best_value:.4f}")
    print('------')
    print("\nBest Parameters:")
    print('------')
    for key, value in study.best_params.items():
        print(f" -> {key}: {value}")
    print('------')
    print(f"\nMin Features to Select: {study.best_trial.params['min_features_to_select']}")
    return study.best_params

