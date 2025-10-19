import torch
import torch.nn as nn

class QPredictorMLP(nn.Module):
    def __init__(
        self,
        input_dim=6,
        hidden_dims=[256, 256, 256, 128, 128, 128, 64, 64, 64, 32],
        dropout=0,
        use_layernorm=True
    ):
        """
        Flexible MLP for Q prediction.

        Args:
            input_dim: Number of input features (default 6)
            hidden_dims: List of hidden layer sizes
            dropout: Dropout rate (applied after each ReLU)
            use_layernorm: Whether to use LayerNorm after each linear
        """
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            if use_layernorm:
                layers.append(nn.LayerNorm(hdim))
            layers.append(nn.ReLU())
            prev_dim = hdim

        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(prev_dim, 1))  # Output: scalar Q
        layers.append(nn.Softplus())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)  # Output shape: [batch_size]