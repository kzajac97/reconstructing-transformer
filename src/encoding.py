import math
from functools import cached_property


class PositionalEncoder(torch.nn.Module):
    """
    Positional encoder used to encode position of the patch in the image or element in sequence.
    Based on the implementation provided by PyTorch: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int = 16, max_len: int = 50, dropout: float = 0.0, device: str = "cpu"):
        """
        :param d_model: dimension of the transformer model
        :param max_len: maximum length of the sequence supported by encoding module
        :param dropout: probability of an element to be dropped out
        """
        super().__init__()

        self.d_model = d_model
        self.max_len = max_len
        self.device = device
        self.registered: bool = False

        self.dropout = torch.nn.Dropout(p=dropout).to(device)

    @cached_property
    def positional_encoding(self) -> torch.Tensor:
        position = torch.arange(self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))
        encoding = torch.zeros(self.max_len, self.d_model)

        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding.to(self.device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.positional_encoding[: x.size(1), :]
        return self.dropout(x)
