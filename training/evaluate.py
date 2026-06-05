import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
) -> dict[str, float]:
    """Compute loss and accuracy on a dataset."""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += images.size(0)

    if total == 0:
        return {"loss": 0.0, "accuracy": 0.0}

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }
