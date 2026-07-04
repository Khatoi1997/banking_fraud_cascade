import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

def get_xgboost_classifier(scale_pos_weight=4.0):
    """
    Initializes a production-grade XGBoost classifier.
    Optimized for tabular cross-domain behavioral data.
    """
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

def get_random_forest_classifier(scale_pos_weight=4.0):
    """
    Alternative model option: Random Forest Classifier.
    Useful for secondary ensemble or baseline benchmarking.
    """
    # Using balanced class weight to mirror XGBoost's scale_pos_weight
    class_weight = {0: 1.0, 1: scale_pos_weight}
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )