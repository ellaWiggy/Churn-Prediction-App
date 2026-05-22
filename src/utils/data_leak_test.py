import os
import sys
import numpy as np
import pandas as pd


from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone

def single_feature_test(X_train, X_test, y_train, y_test, top_feature, threshold=0.90):
    test_model = DecisionTreeClassifier(max_depth=3, random_state=42)

    X_train_single = X_train[[top_feature]]
    X_test_single = X_test[[top_feature]]
    
    test_model.fit(X_train_single, y_train)
    single_auc = roc_auc_score(y_test, test_model.predict_proba(X_test_single)[:, 1])
    
    status = "Potential Leakage" if single_auc > threshold else "No Leakage"
    
    return {
        "feature": top_feature,
        "auc_score": round(single_auc, 4),
        "status": status
    }

def drop_top_feature_test(pipeline, X_train, X_test, y_train, y_test, top_feature):
    pipeline_copy = clone(pipeline)
    pipeline_copy.fit(X_train, y_train)
    base_roc_auc = roc_auc_score(y_test, pipeline_copy.predict_proba(X_test)[:, 1])

    X_train_drop = X_train.drop(columns=[top_feature])
    X_test_drop = X_test.drop(columns=[top_feature])
    
    pipeline_drop = clone(pipeline)
    pipeline_drop.fit(X_train_drop, y_train)
    dropped_roc_auc = roc_auc_score(y_test, pipeline_drop.predict_proba(X_test_drop)[:, 1])
    
    drop_impact = base_roc_auc - dropped_roc_auc
    
    return {
        "base_roc_auc": round(base_roc_auc, 4),
        "dropped_roc_auc": round(dropped_roc_auc, 4),
        "impact": round(drop_impact, 4),
        "robustness": "Robust" if dropped_roc_auc > 0.85 else "Dependent"
    }

def run_data_leak_tests(pipeline, X_train, X_test, y_train, y_test, top_features):
    results = []
    for feat in top_features:
        single = single_feature_test(X_train, X_test, y_train, y_test, feat)
        results.append(single)
        
    drop_feature_test = drop_top_feature_test(pipeline, X_train, X_test, y_train, y_test, top_features[0])
    
    return {"singular_tests": results, "drop_top_feature_test": drop_feature_test}