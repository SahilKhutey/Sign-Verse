import torch


class ModelLoader:

    @staticmethod
    def load_gesture_model(path):
        model = torch.load(path, map_location="cpu")
        model.eval()
        return model
