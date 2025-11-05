"""Aletheion: Epistemic uncertainty quantification for LLMs.

This package implements the Aletheion architecture described in:
    "Aletheion: Fundamental Epistemology for Artificial Intelligence"

The architecture adds epistemic gates (QÅ, QÇ) to transformers, enabling
uncertainty-aware predictions and improved calibration.

Main components:
    - AletheionTransformer: Transformer with epistemic gating (Level 1)
    - AletheionPyramidalTransformer: Transformer with pyramidal epistemology
    - VaroLoss: Variance-Adjusted Ranking Optimization loss
    - PyramidalVAROLoss: Pyramidal VARO loss
    - LocalUncertaintyGate (QÅ): Local evidence quality estimation
    - CrossContextGate (QÇ): Cross-context consensus estimation
    - PyramidalEpistemicGates: 5-vertex pyramidal architecture
"""

from .gates import LocalUncertaintyGate, CrossContextGate, epistemic_softmax
from .loss import VaroLoss, PyramidalVAROLoss
from .model import AletheionTransformer, AletheionModelOutput
from .pyramid import PyramidalEpistemicGates, PyramidalTemperatureModulator, compute_pyramidal_metrics
from .pyramidal_model import AletheionPyramidalTransformer, PyramidalModelOutput

__all__ = [
    "AletheionTransformer",
    "AletheionModelOutput",
    "AletheionPyramidalTransformer",
    "PyramidalModelOutput",
    "LocalUncertaintyGate",
    "CrossContextGate",
    "epistemic_softmax",
    "VaroLoss",
    "PyramidalVAROLoss",
    "PyramidalEpistemicGates",
    "PyramidalTemperatureModulator",
    "compute_pyramidal_metrics",
]
