import torch


class ImagePatcher(torch.nn.Module):
    """
    Class is used to patch images into slices so it can be processed by transformer model.
    """

    def __init__(self, patch_size: int, stride: int, device: str = "cpu"):
        """
        :param patch_size: Size of the patch to be extracted from the image.
        :param stride: The stride with which the patch is extracted, for grayscale image it should be patch_size / 2
        :param device: Device on which the tensor should be stored,
                       Implementation works only for CPU, but class can cast output to GPU for easier usage.
        """
        super().__init__()

        self.patch_size = patch_size
        self.stride = stride
        self.device = device

    def patchify(self, image: torch.Tensor) -> torch.Tensor:
        patched = image.unfold(0, size=self.patch_size, step=self.stride).unfold(
            1, size=self.patch_size, step=self.stride
        )
        patched = patched.flatten(0, 1).flatten(-2, -1)
        return patched

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """Converts a tensor of shape (BATCH_SIZE, X, Y) to (BATCH_SIZE, X * Y / PATCH_SIZE, PATCH_SIZE)"""
        return torch.stack([self.patchify(image.cpu()) for image in batch]).to(self.device)
