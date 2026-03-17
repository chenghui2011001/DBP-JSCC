"""
Aether-Lite Training

训练相关组件：损失函数、训练脚本等
"""

try:
    from .losses import rate_loss, compute_layered_loss
except ImportError:
    pass

try:
    from .wave_loss import fargan_wave_losses
except ImportError:
    pass

__all__ = [
    'rate_loss',
    'compute_layered_loss',
    'fargan_wave_losses'
]