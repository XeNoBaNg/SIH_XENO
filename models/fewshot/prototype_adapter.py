import random

class FewShotAdapter:
    """
    Stub implementation of a Few-Shot Adapter.
    Stores simple 'prototypes' (class names).
    For now, randomly selects one prototype as the match.
    """

    def __init__(self):
        # Preloaded with dummy prototypes
        self.prototypes = ["pothole", "garbage", "waterlogging", "streetlight_broken"]

    def add_prototype(self, label):
        """Add a new class prototype (later this will be an embedding)."""
        self.prototypes.append(label)

    def predict(self, image):
        """
        Pretend to embed 'image' and match with closest prototype.
        For now, just pick one randomly.
        """
        return {"predicted_label": random.choice(self.prototypes)}


if __name__ == "__main__":
    fewshot = FewShotAdapter()
    print("FewShot Prediction:", fewshot.predict("test.jpg"))
    fewshot.add_prototype("illegal_dumping")
    print("With new prototype:", fewshot.predict("test.jpg"))
