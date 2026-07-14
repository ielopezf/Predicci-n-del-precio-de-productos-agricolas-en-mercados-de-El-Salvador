"""
models_advanced.py — Modelos avanzados (CRISP-DM: Modelado - Etapa 2).

RESERVADO PARA ETAPA 2. Se incluye desde ya para dejar fijada la arquitectura
del proyecto y las técnicas seleccionadas en el perfil de trabajo:

    * XGBoost optimizado -> GridSearchCV / Optuna (rúbrica Etapa 2: hiperparámetros).
    * LSTM               -> series temporales, dependencias de largo plazo.
    * Redes Neuronales   -> relaciones no lineales entre múltiples variables.
    * Ensemble           -> combinación de los anteriores.

No forma parte de la entrega de Etapa 1; se deja como plantilla para no
reestructurar el paquete más adelante.

PLANTILLA: solo firmas y docstrings.
"""

from __future__ import annotations


def optimizar_xgboost(X_train, y_train, metodo: str = "optuna"):
    """Búsqueda de hiperparámetros para XGBoost. RESERVADO ETAPA 2."""
    raise NotImplementedError


def construir_lstm(input_shape, unidades: int = 64):
    """Define la arquitectura de una red LSTM (Keras). RESERVADO ETAPA 2."""
    raise NotImplementedError


def construir_red_neuronal(input_dim: int):
    """Define una red neuronal densa (MLP) con Keras. RESERVADO ETAPA 2."""
    raise NotImplementedError


def construir_ensemble(modelos: list):
    """Combina varios modelos (voting/stacking). RESERVADO ETAPA 2."""
    raise NotImplementedError
