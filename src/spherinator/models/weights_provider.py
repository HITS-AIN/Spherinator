import torch


class WeightsProvider:
    def __init__(self, weight_path):
        self.weight_path = weight_path
        self.weights = None

    def load_weights(self):
        try:
            self.weights = torch.load(self.weight_path)
            print(f"Weights loaded successfully from {self.weight_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")

    def get_weights(self):
        if self.weights is None:
            print("Weights not loaded. Call load_weights() first.")
            return None
        return self.weights
