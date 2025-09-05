import torch

class DeformableDetrDetector:
    """
    Stub implementation of Deformable DETR.
    For now, returns fake bounding boxes with dummy categories.
    Later, we will load a pretrained Deformable DETR model.
    """

    def __init__(self):
        # CPU for dev
        self.device = torch.device("cpu")
        # GPU for deployment:
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, image):
        """
        Simulates object detection.
        Returns a list of bounding boxes with category & score.
        """
        dummy_output = [
            {"bbox": [50, 60, 200, 220], "label": "pothole", "score": 0.82},
            {"bbox": [300, 100, 400, 250], "label": "garbage", "score": 0.76},
        ]
        return dummy_output


if __name__ == "__main__":
    model = DeformableDetrDetector()
    preds = model.predict("test.jpg")
    print("DETR Predictions:", preds)
