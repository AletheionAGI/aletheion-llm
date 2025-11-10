# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2024-2025 Felipe Maya Muniz

"""Aletheion: Epistemic uncertainty quantification for LLMs.

This package implements the Aletheion architecture described in:
    "Aletheion: Fundamental Epistemology for Artificial Intelligence"

The architecture adds epistemic gates (Q1, Q2) to transformers, enabling
uncertainty-aware predictions and improved calibration.

Main components:
    - AletheionTransformer: Transformer with epistemic gating (Level 1)
    - AletheionPyramidalTransformer: Transformer with pyramidal epistemology
    - AletheionPyramidalQ1Q2Transformer: Pyramidal with Q1/Q2/Fractal (complete)
    - VaroLoss: Variance-Adjusted Ranking Optimization loss
    - PyramidalVAROLoss: Pyramidal VARO loss
    - PyramidalVAROLossWithQ1Q2: Complete VARO with Q1/Q2/Fractal
    - LocalUncertaintyGate (Q1): Local evidence quality estimation
    - CrossContextGate (Q2): Cross-context consensus estimation
    - PyramidalEpistemicGates: 5-vertex pyramidal architecture
    - PyramidalEpistemicGatesWithQ1Q2: Complete pyramidal with Q1/Q2/Fractal
"""

from .gates import CrossContextGate, LocalUncertaintyGate, epistemic_softmax
from .loss import PyramidalVAROLoss, VaroLoss
from .model import AletheionModelOutput, AletheionTransformer
from .pyramid import (
    PyramidalEpistemicGates,
    PyramidalTemperatureModulator,
    compute_pyramidal_metrics,
)
from .pyramid_q1q2_fractal import (
    EpistemicMultiHeadAttention,
    PyramidalEpistemicGatesWithQ1Q2,
    PyramidalVAROLossWithQ1Q2,
    compute_pyramidal_q1q2_metrics,
)
from .pyramidal_model import AletheionPyramidalTransformer, PyramidalModelOutput
from .pyramidal_q1q2_model import AletheionPyramidalQ1Q2Transformer, PyramidalQ1Q2ModelOutput

__all__ = [
    "AletheionTransformer",
    "AletheionModelOutput",
    "AletheionPyramidalTransformer",
    "PyramidalModelOutput",
    "AletheionPyramidalQ1Q2Transformer",
    "PyramidalQ1Q2ModelOutput",
    "LocalUncertaintyGate",
    "CrossContextGate",
    "epistemic_softmax",
    "VaroLoss",
    "PyramidalVAROLoss",
    "PyramidalVAROLossWithQ1Q2",
    "PyramidalEpistemicGates",
    "PyramidalEpistemicGatesWithQ1Q2",
    "PyramidalTemperatureModulator",
    "EpistemicMultiHeadAttention",
    "compute_pyramidal_metrics",
    "compute_pyramidal_q1q2_metrics",
]
