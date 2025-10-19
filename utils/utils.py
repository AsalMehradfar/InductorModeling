import torch
from model.q_predictor import QPredictorMLP

def load_model(path, *args, **kwargs):
    model = QPredictorMLP(*args, **kwargs)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    print(f"📦 Model loaded from {path}")
    return model
