"""Classification metrics for the test-set summary row in the results log.

Uses the torchmetrics >= 0.11 task-specific API. The old precision_recall
helper and the untyped f1_score signature were removed from torchmetrics.
"""

from torchmetrics.functional.classification import (
    multiclass_f1_score,
    multiclass_precision,
    multiclass_recall,
)

RESULT_COLUMNS = [
    "model",
    "mode",
    "index",
    "macro_f1",
    "micro_f1",
    "macro_precision",
    "macro_recall",
    "micro_precision",
    "micro_recall",
]


def classification_metrics(preds, target, num_classes=3):
    """Macro and micro F1, precision and recall for integer class labels."""
    preds, target = preds.long(), target.long()
    out = {}
    for avg in ("macro", "micro"):
        kw = {"num_classes": num_classes, "average": avg}
        out[f"{avg}_f1"] = multiclass_f1_score(preds, target, **kw).item()
        out[f"{avg}_precision"] = multiclass_precision(preds, target, **kw).item()
        out[f"{avg}_recall"] = multiclass_recall(preds, target, **kw).item()
    return out
