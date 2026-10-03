def classification_metrics(pairs, class_names):
    """Accuracy, macro-F1 and per-class recall from (predicted, true) pairs.
    Macro-F1 averages over the classes that appear in either column, so a
    window that happens to contain no 'River' images is not penalised for it."""
    if not pairs:
        return {"accuracy": None, "macro_f1": None, "per_class_recall": {}}

    correct = sum(pred == true for pred, true in pairs)
    present = [c for c in class_names if any(c in pair for pair in pairs)]
    f1s, recall = [], {}
    for c in present:
        tp = sum(pred == c and true == c for pred, true in pairs)
        fp = sum(pred == c and true != c for pred, true in pairs)
        fn = sum(pred != c and true == c for pred, true in pairs)
        precision = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * rec / (precision + rec) if precision + rec else 0.0)
        if tp + fn:
            recall[c] = round(rec, 4)

    return {
        "accuracy": round(correct / len(pairs), 4),
        "macro_f1": round(sum(f1s) / len(f1s), 4),
        "per_class_recall": recall,
    }
