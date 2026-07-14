"""
features.py — Ingeniería de características (CRISP-DM: Preparación de datos).

Cubre la parte de "feature engineering" de la rúbrica de Etapa 1. Las
características más informativas para precios agrícolas son temporales (lags,
medias móviles) y de calendario (estacionalidad y temporada de cosecha).

Todas las operaciones se realizan POR PRODUCTO (groupby) para no filtrar
información de una serie a otra.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def agregar_variables_calendario(df):
    """Añade mes, trimestre, semana del año y su codificación cíclica (sin/cos).

    La codificación cíclica evita que el modelo interprete diciembre(12) y
    enero(1) como valores lejanos.
    """
    df = df.copy()
    f = pd.to_datetime(df[config.COL_FECHA])
    df["mes"] = f.dt.month
    df["trimestre"] = f.dt.quarter
    df["semana_anio"] = f.dt.isocalendar().week.astype(int)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)
    df["semana_sin"] = np.sin(2 * np.pi * df["semana_anio"] / 52)
    df["semana_cos"] = np.cos(2 * np.pi * df["semana_anio"] / 52)
    return df


def agregar_lags(df, columna=None, lags=(1, 2, 4, 8)):
    """Crea columnas de rezago del precio (semanas anteriores) por producto."""
    df = df.copy()
    columna = columna or config.COL_PRECIO
    for lag in lags:
        df[f"{columna.lower()}_lag{lag}"] = (
            df.groupby(config.COL_PRODUCTO)[columna].shift(lag))
    return df


def agregar_medias_moviles(df, columna=None, ventanas=(4, 8, 12)):
    """Añade medias y desviaciones móviles del precio (tendencia y volatilidad).

    Se usa shift(1) para que la ventana no incluya la semana actual (evita fuga
    de información del objetivo).
    """
    df = df.copy()
    columna = columna or config.COL_PRECIO
    g = df.groupby(config.COL_PRODUCTO)[columna]
    for v in ventanas:
        df[f"media_movil_{v}"] = g.transform(
            lambda s: s.shift(1).rolling(v, min_periods=1).mean())
        df[f"desv_movil_{v}"] = g.transform(
            lambda s: s.shift(1).rolling(v, min_periods=2).std())
    return df


def marcar_temporada_cosecha(df):
    """Bandera binaria `es_cosecha` según el mes y el cultivo (aporte local).

    Usa `config.MESES_COSECHA` (calendario agrícola salvadoreño). Sustenta el
    criterio "Originalidad e innovación - aporte local".
    """
    df = df.copy()
    mes = pd.to_datetime(df[config.COL_FECHA]).dt.month
    df["es_cosecha"] = [
        1 if m in config.MESES_COSECHA.get(g, []) else 0
        for m, g in zip(mes, df["Grupo"])
    ]
    return df


def construir_matriz_modelado(df, dropna=True):
    """Devuelve (X, y) listos para entrenar.

    - y = precio (variable objetivo).
    - X = variables numéricas + codificación one-hot del producto (Grupo).
    - Se descartan identificadores (Fecha, Producto) y las filas con NaN
      generadas por los lags iniciales.

    El DataFrame se ordena por Fecha para permitir un split temporal correcto.
    """
    df = df.sort_values(config.COL_FECHA).reset_index(drop=True)

    if dropna:
        df = df.dropna().reset_index(drop=True)

    y = df[config.COL_PRECIO].copy()

    quitar = [config.COL_PRECIO, config.COL_FECHA, config.COL_PRODUCTO]
    X = df.drop(columns=[c for c in quitar if c in df.columns])
    # One-hot del grupo de cultivo (categórica)
    if "Grupo" in X.columns:
        X = pd.get_dummies(X, columns=["Grupo"], prefix="grupo")
    return X, y
