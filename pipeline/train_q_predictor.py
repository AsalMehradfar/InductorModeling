import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from model.q_predictor import QPredictorMLP
from pipeline.dataset import QDataset
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.visual_utils import plot_loss_curves

def train_q_predictor(
    train_df,
    val_df,
    mean,
    std,
    output_dir="checkpoints",
    batch_size=16384,
    hidden_dims=[256, 256, 256, 128, 128, 128, 64, 64, 64, 32],
    dropout=0,
    use_layernorm=True,
    lr=1e-3,
    # weight_decay=1e-5,
    max_epochs=300,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    os.makedirs(output_dir, exist_ok=True)

    # Datasets & loaders
    train_dataset = QDataset(train_df, mean, std)
    val_dataset = QDataset(val_df, mean, std)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Model
    model = QPredictorMLP(input_dim=6, hidden_dims=hidden_dims, dropout=dropout, use_layernorm=use_layernorm)
    model.to(device)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    best_model_path = os.path.join(output_dir, "q_predictor.pt")

    train_losses = []
    val_losses = []

    print(f"[INFO] Training on {device}...")

    for epoch in range(max_epochs):
        model.train()
        total_train_loss = 0.0

        for x_batch, y_batch in tqdm(train_loader, desc="Training Batches", leave=False):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(x_batch)

        avg_train_loss = total_train_loss / len(train_dataset)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in tqdm(val_loader, desc="Validation Batches", leave=False):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                y_pred = model(x_batch)
                loss = criterion(y_pred, y_batch)
                total_val_loss += loss.item() * len(x_batch)

        avg_val_loss = total_val_loss / len(val_dataset)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch+1:02d}/{max_epochs} | Train MSE: {avg_train_loss:.4f} | Val MSE: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"    ✅ Saved new best model → {best_model_path}")

    # Plot loss curves
    plot_loss_curves(
        train_losses=train_losses,
        val_losses=val_losses,
        title="Q Predictor Training Loss",
        save_path=os.path.join(output_dir, "q_predictor_loss.pdf"),
        log_scale=True
    )
    return best_model_path
