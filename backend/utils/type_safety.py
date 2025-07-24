def to_python_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif hasattr(obj, "item") and callable(obj.item):
        # For np.float64, np.int64, etc.
        return obj.item()
    return obj

def to_podium(preds, n=None):
    """
    Convert a list of prediction dicts to a podium list.
    If n is None, use the length of preds.
    """
    if n is None:
        n = len(preds)
    podium = []
    for i, pred in enumerate(preds[:n]):
        podium.append({
            "position": i + 1,
            "driver": pred.get('name', pred.get('driver', 'Unknown')),
            "team": pred.get('team', 'Unknown'),
            "probability": pred.get('win_probability', pred.get('probability', 0.0))
        })
    return podium 