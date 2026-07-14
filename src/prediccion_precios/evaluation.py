"""
evaluation.py — Métricas y validación (CRISP-DM: Evaluación).

Cubre el requisito de Etapa 1: "Métricas iniciales y comparación" y la tabla
comparativa de modelos baseline. Implementa las métricas del perfil (MAE, RMSE,
MAPE, R²), el split temporal y la validación cruzada temporal.

Las métricas están en NumPy puro (sin dependencias externas) para ser
reproducibles y ligeras. El split respeta el orden temporal (NO aleatorio),
requisito indispensable en series de tiempo para evitar fuga de información.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def calcular_metricas(y_true, y_pred):
    """Devuelve {'MAE', 'RMSE', 'MAPE_%', 'R2'} para un conjunto de predicciones.

    MAPE se reporta en porcentaje (%). La meta del objetivo general es
    MAPE < 15% (config.META_MAPE_OBJETIVO = 0.15).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_true - y_pred
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    # MAPE ignorando divisiones por cero (precios nunca deberían ser 0)
    mask = y_true != 0
    mape = np.mean(np.abs(err[mask] / y_true[mask])) * 100

    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4),
            "MAPE_%": round(mape, 4), "R2": round(r2, 4)}


# --------------------------------------------------------------------------- #
# Split temporal
# --------------------------------------------------------------------------- #
def split_temporal(X, y, test_size=config.TEST_SIZE):
    """Divide en train/test respetando el orden (las últimas filas -> test).

    Se asume que X, y vienen ordenados temporalmente (así los entrega
    features.construir_matriz_modelado). Devuelve X_train, X_test, y_train, y_test.
    """
    n = len(X)
    n_test = int(np.ceil(n * test_size))
    n_train = n - n_test
    X_train = X.iloc[:n_train] if hasattr(X, "iloc") else X[:n_train]
    X_test = X.iloc[n_train:] if hasattr(X, "iloc") else X[n_train:]
    y_train = y.iloc[:n_train] if hasattr(y, "iloc") else y[:n_train]
    y_test = y.iloc[n_train:] if hasattr(y, "iloc") else y[n_train:]
    return X_train, X_test, y_train, y_test


# --------------------------------------------------------------------------- #
# Tabla comparativa
# --------------------------------------------------------------------------- #
def tabla_comparativa(resultados):
    """Construye la tabla comparativa de modelos exigida por la rúbrica.

    resultados : {nombre_modelo: {'MAE':..., 'RMSE':..., 'MAPE_%':..., 'R2':...}}
    Devuelve un DataFrame ordenado por MAPE ascendente (mejor arriba).
    """
    tabla = pd.DataFrame(resultados).T
    if "MAPE_%" in tabla.columns:
        tabla = tabla.sort_values("MAPE_%")
    tabla.index.name = "Modelo"
    return tabla


# --------------------------------------------------------------------------- #
# Validación cruzada temporal
# --------------------------------------------------------------------------- #
def _folds_temporales(n, n_splits):
    """Genera índices (train, test) tipo TimeSeriesSplit (ventana expansiva)."""
    tam = n // (n_splits + 1)
    for i in range(1, n_splits + 1):
        fin_train = tam * i
        fin_test = tam * (i + 1) if i < n_splits else n
        yield np.arange(0, fin_train), np.arange(fin_train, fin_test)


def validacion_cruzada_temporal(modelo_fn, X, y, n_splits=config.CV_SPLITS):
    """Validación cruzada temporal (ventana expansiva).

    Parameters
    ----------
    modelo_fn : callable
        Función SIN argumentos que devuelve un modelo nuevo sin entrenar
        (p. ej. `lambda: models_baseline.nuevo_xgboost()`). Se re-crea en cada
        fold para no arrastrar entrenamiento previo.
    X, y : datos ordenados temporalmente.

    Returns: dict con media y desviación de cada métrica entre folds.
    """
    Xv = X.values if hasattr(X, "values") else np.asarray(X)
    yv = y.values if hasattr(y, "values") else np.asarray(y)

    acum = {"MAE": [], "RMSE": [], "MAPE_%": [], "R2": []}
    for idx_tr, idx_te in _folds_temporales(len(Xv), n_splits):
        modelo = modelo_fn()
        modelo.fit(Xv[idx_tr], yv[idx_tr])
        pred = modelo.predict(Xv[idx_te])
        for k, v in calcular_metricas(yv[idx_te], pred).items():
            acum[k].append(v)

    return {k: {"media": round(float(np.mean(v)), 4),
                "std": round(float(np.std(v)), 4)} for k, v in acum.items()}
