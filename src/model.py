import torch

from encoding import PositionalEncoder
from patcher import ImagePatcher


class MiniVisionTransformerClassifier(torch.nn.Module):
    """
    Mini Vision Transformer classifier is a minimal auto-didactic implementation of the Vision Transformer model
    based on the paper: https://arxiv.org/abs/2010.11929
    """

    def __init__(
        self,
        model_dim: int,
        patch_size: int,
        patches_in_image: int,
        num_classes: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        device: str = "cuda",
    ):
        """
        :param model_dim: dimension of the transformer model
        :param patch_size: size of the patch to be extracted from the image
        :param patches_in_image: number of patches in the image
        :param num_classes: number of classes in the dataset
        :param num_heads: number of heads in the transformer model
        :param num_layers: number of layers in the encoder stack of the transformer model
        :param dim_feedforward: dimension of the feedforward network in the transformer model
        """
        super().__init__()

        self.patcher = ImagePatcher(patch_size=patch_size, stride=patch_size, device=device)
        self.embedding = torch.nn.Linear(in_features=model_dim, out_features=model_dim, bias=False).to(device)
        self.positional_encoding = PositionalEncoder(d_model=model_dim, max_len=1 + patches_in_image, device=device)
        self.pooler = torch.nn.Linear(in_features=patches_in_image * model_dim, out_features=model_dim).to(device)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model=model_dim, nhead=num_heads, dim_feedforward=dim_feedforward, batch_first=True
            ),
            num_layers=num_layers,
        ).to(device)

        self.classifier = torch.nn.Linear(in_features=model_dim, out_features=num_classes).to(device)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        patches = self.patcher(batch)
        embeddings = self.embedding(patches)
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.encoder(embeddings)

        outputs = self.pooler(embeddings.flatten(start_dim=1))
        logits = self.classifier(outputs)
        predictions = torch.nn.functional.softmax(logits, dim=-1)

        return predictions
