def fuse_predictions(predictions_list):
    """
    Simple fusion function.
    predictions_list: list of lists of predictions from detectors.
      Example: [[{bbox:..., label:...}, {...}], [{bbox:..., label:...}]]
    Returns merged list for now.
    """

    merged = []
    for preds in predictions_list:
        merged.extend(preds)

    # Later we can add NMS (non-max suppression) and confidence weighting
    return {"fused_predictions": merged}
    

if __name__ == "__main__":
    dummy1 = [{"bbox": [0, 0, 100, 100], "label": "pothole", "score": 0.8}]
    dummy2 = [{"bbox": [20, 20, 120, 120], "label": "garbage", "score": 0.7}]
    print(fuse_predictions([dummy1, dummy2]))
