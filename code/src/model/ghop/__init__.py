"""
GHOP (Generative Hand-Object Prior) integration for HOLD.
"""

from .ghop_prior import GHOPPriorModule, load_ghop_prior
from .ghop_loss import GHOPSDSLoss
from .text_template import Obj2Text, create_text_template  # CHANGED: TextTemplateGenerator -> Obj2Text
from .hand_field import HandSkeletalField

__all__ = [
    'GHOPPriorModule',
    'load_ghop_prior',
    'GHOPSDSLoss',
    'Obj2Text',              # CHANGED: TextTemplateGenerator -> Obj2Text
    'create_text_template',
    'HandSkeletalField',
]
