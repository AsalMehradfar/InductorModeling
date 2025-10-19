from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader
from pipeline.dataset import QDataset
import numpy as np
from tqdm import tqdm
import torch

def evaluate_regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    print("\n📊 Regression Metrics on Test Set:")
    print(f"   MAE  (Mean Abs Error)        : {mae:.6f}")
    print(f"   MSE  (Mean Sq Error)         : {mse:.6f}")
    print(f"   RMSE (Root Mean Sq Error)    : {rmse:.6f}")
    print(f"   R²   (R-squared Score)       : {r2:.6f}")
    print(f"   MAPE (Mean Abs % Error)      : {mape:.2f}%")

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mape": mape,
    }

def evaluate_model_on_dataframe(model, df, mean, std, input_columns, target_column="Q", device="cpu"):

    # Datasets & loaders
    test_dataset = QDataset(df, mean, std)
    test_loader = DataLoader(test_dataset, batch_size=16384, shuffle=False, num_workers=4)

    model.eval()
    y_pred_all = []
    y_true_all = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(test_loader, desc="Testing Batches", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            y_pred_all.extend(list(y_pred.cpu().detach().numpy()))
            y_true_all.extend(list(y_batch.cpu().detach().numpy()))
    
    test_metrics = evaluate_regression_metrics(np.array(y_true_all), np.array(y_pred_all))
    test_metrics = {k: float(v) for k, v in test_metrics.items()}

    return test_metrics
