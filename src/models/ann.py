# Feed-forward baseline over a pooled input vector; no notion of word order.

from torch import nn


class ANNClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list,
        num_classes: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers += [nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
