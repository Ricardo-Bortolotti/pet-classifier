from dataclasses import dataclass
from enum import StrEnum


class FreezeStrategy(StrEnum):
    HEAD_ONLY = "head_only"
    PARTIAL = "partial"
    FULL = "full"


@dataclass
class ModelConfig:
    """Configuration for a single model architecture."""

    name: str
    num_classes: int = 2
    pretrained: bool = True
    freeze_strategy: FreezeStrategy = FreezeStrategy.FULL
    dropout: float = 0.2
    image_size: int = 224

    @property
    def freeze_backbone(self) -> bool:
        return self.freeze_strategy == FreezeStrategy.HEAD_ONLY
