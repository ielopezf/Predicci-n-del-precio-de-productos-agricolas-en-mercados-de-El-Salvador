"""
eda.py — Análisis Exploratorio de Datos (CRISP-DM: Comprensión de datos).

Cubre el requisito de Etapa 1: "EDA completo (gráficos obligatorios)". Cada
función de graficado guarda la figura en `config.FIGURES_DIR` (para incrustarla
en el reporte) y devuelve la ruta del archivo generado.

Usa el backend "Agg" de matplotlib (no interactivo) para poder ejecutarse tanto
en notebooks como en scripts/servidores sin display.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from . import config

sns.set_theme(style="whitegrid")


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _slug(texto: str) -> str:
    """Convierte un nombre de producto en un nombre de archivo seguro."""
    t = unicodedata.normalize("NFKD", str(texto))
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^A-Za-z0-9]+", "_", t).strip("_").lower()
    return t or "figura"


def _guardar(fig, nombre: str) -> Path:
    """Guarda la figura en FIGURES_DIR y la cierra. Devuelve la ruta."""
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ruta = config.FIGURES_DIR / nombre
    fig.savefig(ruta, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return ruta


def pivot_precios(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla ancha: índice=Fecha, una columna de precio por producto."""
    return df.pivot_table(index=config.COL_FECHA, columns=config.COL_PRODUCTO,
                          values=config.COL_PRECIO, aggfunc="mean").sort_index()


# --------------------------------------------------------------------------- #
# 1. Estadística descriptiva
# --------------------------------------------------------------------------- #
def resumen_estadistico(df: pd.DataFrame) -> pd.DataFrame:
    """Estadística descriptiva del precio por producto + % de faltantes."""
    g = df.groupby(config.COL_PRODUCTO)[config.COL_PRECIO]
    resumen = g.describe().round(3)
    resumen["pct_faltantes"] = (g.apply(lambda s: s.isna().mean() * 100)).round(2)
    return resumen


# --------------------------------------------------------------------------- #
# 2. Series temporales (tendencia)
# --------------------------------------------------------------------------- #
def graficar_serie_temporal(df: pd.DataFrame, producto: str, guardar: bool = True) -> Path | None:
    """Serie de tiempo del precio de un producto (tendencia general)."""
    sub = df[df[config.COL_PRODUCTO] == producto].sort_values(config.COL_FECHA)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(sub[config.COL_FECHA], sub[config.COL_PRECIO], lw=1.2)
    ax.set_title(f"Evolución del precio — {producto}")
    ax.set_xlabel("Fecha"); ax.set_ylabel("Precio (US$)")
    return _guardar(fig, f"serie_{_slug(producto)}.png") if guardar else None


# --------------------------------------------------------------------------- #
# 3. Estacionalidad
# --------------------------------------------------------------------------- #
def graficar_estacionalidad(df: pd.DataFrame, producto: str, guardar: bool = True) -> Path | None:
    """Boxplot de precio por mes para visualizar patrones estacionales."""
    sub = df[df[config.COL_PRODUCTO] == producto].copy()
    sub["Mes"] = pd.to_datetime(sub[config.COL_FECHA]).dt.month
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.boxplot(data=sub, x="Mes", y=config.COL_PRECIO, ax=ax)
    ax.set_title(f"Estacionalidad mensual del precio — {producto}")
    ax.set_xlabel("Mes"); ax.set_ylabel("Precio (US$)")
    return _guardar(fig, f"estacionalidad_{_slug(producto)}.png") if guardar else None


# --------------------------------------------------------------------------- #
# 4. Distribución
# --------------------------------------------------------------------------- #
def graficar_distribucion(df: pd.DataFrame, guardar: bool = True) -> Path | None:
    """Histograma/KDE del precio por producto (facetas)."""
    productos = df[config.COL_PRODUCTO].unique()
    n = len(productos)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 3.2 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, prod in zip(axes, productos):
        datos = df.loc[df[config.COL_PRODUCTO] == prod, config.COL_PRECIO].dropna()
        sns.histplot(datos, kde=True, ax=ax, bins=30)
        ax.set_title(prod, fontsize=9); ax.set_xlabel("Precio (US$)")
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Distribución del precio por producto", y=1.02)
    return _guardar(fig, "distribucion_precios.png") if guardar else None


# --------------------------------------------------------------------------- #
# 5. Correlación
# --------------------------------------------------------------------------- #
def matriz_correlacion(df: pd.DataFrame, columnas=None, guardar: bool = True) -> pd.DataFrame:
    """Heatmap de correlación.

    - Si `df` ya trae variables numéricas integradas (precio + combustible + etc.),
      correlaciona esas columnas (pasar `columnas`).
    - Si solo trae precios en formato largo, correlaciona los precios entre
      productos (usando `pivot_precios`).

    Returns
    -------
    pd.DataFrame con la matriz de correlación.
    """
    if columnas is not None:
        base = df[columnas]
        titulo = "Correlación entre variables"
        nombre = "correlacion_variables.png"
    else:
        base = pivot_precios(df)
        titulo = "Correlación de precios entre productos"
        nombre = "correlacion_productos.png"

    corr = base.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 3, 1.0 * len(corr) + 2))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title(titulo)
    if guardar:
        _guardar(fig, nombre)
    else:
        plt.close(fig)
    return corr.round(3)


# --------------------------------------------------------------------------- #
# 6. Outliers
# --------------------------------------------------------------------------- #
def detectar_outliers(df: pd.DataFrame, metodo: str = "iqr", umbral: float = 1.5) -> pd.DataFrame:
    """Detecta valores atípicos del precio por producto.

    metodo : "iqr" (rango intercuartílico) o "zscore".

    Returns
    -------
    pd.DataFrame resumen por producto: n, n_outliers, pct_outliers, limites.
    """
    filas = []
    for prod, sub in df.groupby(config.COL_PRODUCTO):
        s = sub[config.COL_PRECIO].dropna()
        if s.empty:
            continue
        if metodo == "iqr":
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - umbral * iqr, q3 + umbral * iqr
        elif metodo == "zscore":
            mu, sd = s.mean(), s.std(ddof=0)
            low, high = mu - 3 * sd, mu + 3 * sd
        else:
            raise ValueError("metodo debe ser 'iqr' o 'zscore'")
        out = ((s < low) | (s > high)).sum()
        filas.append({
            "Producto": prod, "n": len(s), "n_outliers": int(out),
            "pct_outliers": round(out / len(s) * 100, 2),
            "limite_inf": round(low, 3), "limite_sup": round(high, 3),
        })
    return pd.DataFrame(filas).sort_values("pct_outliers", ascending=False).reset_index(drop=True)
