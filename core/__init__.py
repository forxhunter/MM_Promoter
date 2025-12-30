"""
Core Markovian Promoter Framework

This package provides the fundamental classes and utilities for implementing
Markovian promoter models in gene regulatory networks.

Classes:
--------
Core Framework for Markovian Gene Regulation Models

This package provides the foundational classes for modeling gene regulation
using explicit Markovian dynamics for promoter states.

Key Classes:
    - MarkovianPromoterModel: Single-input promoter with multi-site binding
    - MultiInputPromoter: Multi-input promoter with combinatorial logic
    - PromoterStateTracker: Utility for tracking promoter state transitions
"""

from .markovian_promoter import MarkovianPromoterModel, PromoterStateTracker
from .multi_input_promoter import MultiInputPromoter

__all__ = [
    'MarkovianPromoterModel',
    'MultiInputPromoter',
    'PromoterStateTracker',
]

__version__ = '1.0.0'
