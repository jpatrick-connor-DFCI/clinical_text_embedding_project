"""Standalone first-event landmark survival modeling from precomputed notes."""

from .config import Config
from .model import NoteSurvivalTransformer
from .losses import sampled_cox_loss, full_breslow_loss

__version__ = "0.1.0"
__all__ = ["Config", "NoteSurvivalTransformer", "sampled_cox_loss", "full_breslow_loss"]
