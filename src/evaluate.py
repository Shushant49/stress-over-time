"""Metrics shared by every model, so baseline and transformer are scored the same way."""
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        # macro = F1 for each class separately, then the plain average, so both classes count equally
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }
