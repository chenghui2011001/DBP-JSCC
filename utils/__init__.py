"""Utility package for DBP-JSCC.

Keep package import side effects minimal so lightweight scripts such as
dataset preparation can run without importing the full training stack.
"""

__all__ = [
    "feature_extraction",
    "feature_spec",
    "channel_sim",
    "audio_visualizer",
    "real_data_loader",
    "metrics",
]
