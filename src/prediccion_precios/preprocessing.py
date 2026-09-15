"""
preprocessing.py — Limpieza e integración (CRISP-DM: Preparación de datos).

Cubre parte del requisito de Etapa 1: "Preprocesamiento y feature engineering".
Flujo típico (ver notebook 02):

    limpiar_precios -> tratar_outliers -> resamplear_frecuencia (semanal)
    -> tratar_faltantes -> integrar_fuentes -> (features) -> guardar_processed

Todas las operaciones se hacen POR PRODUCTO para no mezclar series distintas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# 1. Limpieza
# --------------------------------------------------------------------------- #
def limpiar_precios(df):
    """Normaliza nombres, ordena por producto/fecha y elimina duplicados exactos.

    Colapsa además registros duplicados (mismo producto y fecha) promediando el
    precio, para garantizar una sola observación por producto-día.
    """
    df = df.copy()
    df[config.COL_PRODUCTO] = df[config.COL_PRODUCTO].astype(str).str.strip()
    df[config.COL_FECHA] = pd.to_datetime(df[config.COL_FECHA], errors="coerce")
    df = df.dropna(subset=[config.COL_FECHA])
    # Un solo precio por producto-día
    df = (df.groupby([config.COL_PRODUCTO, "Grupo", config.COL_FECHA], as_index=False)
            [config.COL_PRECIO].mean())
    return df.sort_values([config.COL_PRODUCTO, config.COL_FECHA]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 2. Outliers
# --------------------------------------------------------------------------- #
def tratar_outliers(df, metodo="clip_iqr", umbral=1.5):
    """Recorta (winsoriza) los outliers del precio por producto.

    metodo="clip_iqr": limita el precio al rango [Q1 - k*IQR, Q3 + k*IQR] de cada
    producto. Evita eliminar filas (importante en series de tiempo).
    """
    if metodo != "clip_iqr":
        raise ValueError("Solo se implementa metodo='clip_iqr'.")
    df = df.copy()

    def _clip(s):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return s.clip(q1 - umbral * iqr, q3 + umbral * iqr)

    df[config.COL_PRECIO] = df.groupby(config.COL_PRODUCTO)[config.COL_PRECIO].transform(_clip)
    return df


# --------------------------------------------------------------------------- #
# 3. Resampleo a frecuencia común (semanal)
# --------------------------------------------------------------------------- #
def resamplear_frecuencia(df, freq=config.FRECUENCIA):
    """Homogeneiza la serie de precios a frecuencia fija (semanal por defecto).

    Devuelve un DataFrame largo [Fecha, Producto, Grupo, Precio] con una fila por
    producto y semana (media de los días de esa semana).
    """
    partes = []
    for (prod, grupo), sub in df.groupby([config.COL_PRODUCTO, "Grupo"]):
        s = (sub.set_index(config.COL_FECHA)[config.COL_PRECIO]
                .sort_index().resample(freq).mean())
        w = s.to_frame(config.COL_PRECIO)
        w[config.COL_PRODUCTO] = prod
        w["Grupo"] = grupo
        partes.append(w.reset_index())
    out = pd.concat(partes, ignore_index=True)
    return out.sort_values([config.COL_PRODUCTO, config.COL_FECHA]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 4. Faltantes
# --------------------------------------------------------------------------- #
def diagnosticar_nulos(df, etiqueta=""):
    """Tabla de nulos por columna (conteo y %), para documentar cada etapa.

    Se usa antes/después de cada paso de tratamiento (resampleo, interpolación,
    integración de fuentes, features) para dejar evidencia de qué se corrigió.
    """
    n = df.isna().sum()
    n = n[n > 0]
    tabla = pd.DataFrame({"n_nulos": n, "pct_nulos": (n / len(df) * 100).round(2)})
    tabla = tabla.sort_values("n_nulos", ascending=False)
    if etiqueta:
        print(f"--- Nulos: {etiqueta} ({len(df)} filas) ---")
    print(tabla if not tabla.empty else "Sin nulos.")
    return tabla


def tratar_faltantes(df, freq=config.FRECUENCIA):
    """Reindexa cada producto al rango semanal completo e interpola el precio.

    Garantiza una rejilla temporal continua (sin huecos) por producto, requisito
    para calcular lags/medias móviles correctamente.
    """
    inicio, fin = df[config.COL_FECHA].min(), df[config.COL_FECHA].max()
    rejilla = pd.date_range(inicio, fin, freq=freq)
    partes = []
    for (prod, grupo), sub in df.groupby([config.COL_PRODUCTO, "Grupo"]):
        s = sub.set_index(config.COL_FECHA)[config.COL_PRECIO].sort_index()
        s = s.reindex(rejilla).interpolate(method="linear", limit_direction="both")
        w = s.to_frame(config.COL_PRECIO)
        w[config.COL_PRODUCTO] = prod
        w["Grupo"] = grupo
        w.index.name = config.COL_FECHA
        partes.append(w.reset_index())
    out = pd.concat(partes, ignore_index=True)
    return out.sort_values([config.COL_PRODUCTO, config.COL_FECHA]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5. Integración de fuentes externas
# --------------------------------------------------------------------------- #
def integrar_fuentes(df_precios, combustible=None, precipitacion=None,
                     ipc=None, produccion=None, freq=config.FRECUENCIA):
    """Une los precios semanales con las fuentes externas alineando por tiempo.

    - combustible / precipitacion: se agregan a nivel semanal (misma rejilla 'W').
    - ipc: mensual -> se une por (Anio, Mes) de cada semana.
    - produccion: anual -> se une por (Anio, Grupo).

    Las columnas externas faltantes en los extremos se rellenan por ffill/bfill.
    """
    df = df_precios.copy()
    df["Anio"] = df[config.COL_FECHA].dt.year
    df["Mes"] = df[config.COL_FECHA].dt.month

    if combustible is not None:
        comb = (combustible.set_index(config.COL_FECHA).sort_index()
                .resample(freq).mean().reset_index())
        df = df.merge(comb, on=config.COL_FECHA, how="left")

    if precipitacion is not None:
        prec = (precipitacion.set_index(config.COL_FECHA).sort_index()
                .resample(freq).agg({"precipitacion_mm": "sum", "lluvia_max_mm": "max"})
                .reset_index())
        df = df.merge(prec, on=config.COL_FECHA, how="left")

    if ipc is not None:
        ipc_m = ipc[["Anio", "Mes", "ipc"]].rename(columns={"ipc": "ipc_transporte"})
        df = df.merge(ipc_m, on=["Anio", "Mes"], how="left")

    if produccion is not None:
        df = df.merge(produccion, on=["Anio", "Grupo"], how="left")

    # Relleno de extremos en columnas externas (por producto, en orden temporal)
    externas = [c for c in df.columns
                if c not in (config.COL_FECHA, config.COL_PRODUCTO, "Grupo",
                             config.COL_PRECIO, "Anio", "Mes")]
    df = df.sort_values([config.COL_PRODUCTO, config.COL_FECHA])
    df[externas] = (df.groupby(config.COL_PRODUCTO)[externas]
                      .transform(lambda g: g.ffill().bfill()))
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 6. Persistencia
# --------------------------------------------------------------------------- #
def guardar_processed(df, ruta=None):
    """Guarda el dataset final en data/processed/ como CSV (sin dependencias extra)."""
    ruta = ruta or config.DATASET_MODELADO
    ruta.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ruta, index=False)
    return ruta
