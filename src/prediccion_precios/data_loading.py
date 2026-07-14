"""
data_loading.py — Carga y descripción de datos (CRISP-DM: Comprensión de datos).

Cubre el requisito de Etapa 1: "Recolección y descripción del dataset
(fuente, tamaño, variables, licencia)". Lectores para la fuente principal
(precios MAG consolidados) y para cada fuente externa aportada por el equipo,
dejándolas en formato limpio y homogéneo (columna Fecha + valores).

Fuentes:
  * precios_consolidados.xlsx  — MAG, diario (días hábiles), 2021-2024.
  * precios_combustibles_sv    — DGEHM, diario, por región y tipo de combustible.
  * precipitaciones            — diario, 2021-2024, mm promedio/mediana/máx/total.
  * IPC                        — mensual, formato ancho (año x mes).
  * Produccion.csv             — FAOSTAT, anual, por cultivo (área/rendimiento/producción).
"""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

from . import config


# --------------------------------------------------------------------------- #
# Utilidades
# --------------------------------------------------------------------------- #
def _normalizar(texto):
    """Mayúsculas sin acentos, para comparar nombres de producto de forma flexible."""
    if texto is None:
        return ""
    texto = unicodedata.normalize("NFKD", str(texto))
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip().upper()


def grupo_cultivo(producto):
    """Grupo de cultivo (MAÍZ/FRIJOL/ARROZ/TOMATE/NARANJA) de un producto.

    Permite unir precios detallados con la producción anual FAOSTAT (por cultivo).
    """
    n = _normalizar(producto)
    for grupo in ("MAIZ", "FRIJOL", "ARROZ", "TOMATE", "NARANJA"):
        if grupo in n:
            return "MAÍZ" if grupo == "MAIZ" else grupo
    return None


# --------------------------------------------------------------------------- #
# Fuente principal: precios MAG
# --------------------------------------------------------------------------- #
def cargar_precios_consolidados(ruta=None, hoja=config.HOJA_PRECIOS):
    """Carga el consolidado del MAG y normaliza tipos.

    Returns: DataFrame [Archivo, Hoja, Producto, Precio, Fecha, Dia, Mes, Anio, Grupo].
    """
    ruta = ruta or config.PRECIOS_CONSOLIDADOS
    df = pd.read_excel(ruta, sheet_name=hoja)
    df[config.COL_FECHA] = pd.to_datetime(df[config.COL_FECHA], errors="coerce")
    df[config.COL_PRECIO] = pd.to_numeric(df[config.COL_PRECIO], errors="coerce")
    df[config.COL_PRODUCTO] = df[config.COL_PRODUCTO].astype(str).str.strip()
    df["Grupo"] = df[config.COL_PRODUCTO].map(grupo_cultivo)
    return df.sort_values([config.COL_PRODUCTO, config.COL_FECHA]).reset_index(drop=True)


def filtrar_productos_objetivo(df, productos=None):
    """Filtra a los productos objetivo (config.PRODUCTOS_OBJETIVO por defecto)."""
    productos = productos or config.PRODUCTOS_OBJETIVO
    objetivo = {_normalizar(p) for p in productos}
    mask = df[config.COL_PRODUCTO].map(lambda p: _normalizar(p) in objetivo)
    return df[mask].reset_index(drop=True)


def describir_dataset(df):
    """Ficha descriptiva del dataset para el documento de Etapa 1."""
    fechas = df[config.COL_FECHA].dropna()
    faltantes = (df.isna().mean() * 100).round(2).to_dict()
    return {
        "n_filas": len(df),
        "n_columnas": df.shape[1],
        "columnas": list(df.columns),
        "rango_fechas": (fechas.min(), fechas.max()) if len(fechas) else (None, None),
        "n_fechas_unicas": fechas.dt.date.nunique(),
        "n_productos": df[config.COL_PRODUCTO].nunique(),
        "productos": sorted(df[config.COL_PRODUCTO].unique().tolist()),
        "conteo_por_categoria": df[config.COL_CATEGORIA].value_counts().to_dict(),
        "pct_faltantes_por_columna": faltantes,
        "precio_describe": df[config.COL_PRECIO].describe().round(3).to_dict(),
    }


# --------------------------------------------------------------------------- #
# Fuentes externas
# --------------------------------------------------------------------------- #
def cargar_combustible(ruta=None, anios=(2021, 2022, 2023, 2024)):
    """Precios de combustible resumidos a series nacionales diarias.

    Notas del dato real:
      * Las columnas 'Diesel *' están en 0 para 2021-2024; el diesel vigente es
        el bajo azufre ('Diesel Azufre *'), usado aquí como `diesel`.
      * Hay fechas con dos registros (días de ajuste); se promedian por fecha.

    Returns: DataFrame [Fecha, diesel, gas_regular, gas_especial] (una fila/día).
    """
    ruta = ruta or config.FUENTE_COMBUSTIBLE
    df = pd.read_excel(ruta)
    df = df[df["Año"].isin(anios)].copy()
    df[config.COL_FECHA] = pd.to_datetime(
        dict(year=df["Año"], month=df["Mes"], day=df["Día"]), errors="coerce"
    )
    grupos = {
        "diesel": ["Diesel Azufre Central", "Diesel Azufre Occidental", "Diesel Azufre Oriental"],
        "gas_regular": ["Gas Regular Central", "Gas Regular Occidental", "Gas Regular Oriental"],
        "gas_especial": ["Gas Especial Central", "Gas Especial Occidental", "Gas Especial Oriental"],
    }
    out = pd.DataFrame({config.COL_FECHA: df[config.COL_FECHA]})
    for nombre, cols in grupos.items():
        out[nombre] = df[cols].replace(0, np.nan).mean(axis=1)
    out = out.dropna(subset=[config.COL_FECHA])
    out = out.groupby(config.COL_FECHA, as_index=False).mean()
    return out.sort_values(config.COL_FECHA).reset_index(drop=True)


def cargar_precipitacion(ruta=None):
    """Precipitaciones diarias nacionales.

    Returns: DataFrame [Fecha, precipitacion_mm, lluvia_max_mm].
    """
    ruta = ruta or config.FUENTE_PRECIPITACION
    df = pd.read_excel(ruta)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce")
    out = df[["Fecha", "Precipitación_promedio_mm", "Precipitación_máxima_mm"]].copy()
    out.columns = [config.COL_FECHA, "precipitacion_mm", "lluvia_max_mm"]
    return out.sort_values(config.COL_FECHA).reset_index(drop=True)


def cargar_ipc(ruta=None):
    """IPC mensual '1.7 Transporte' (Índice Dic 2009=100), proxy del costo de transporte.

    Se lee desde un CSV limpio (`ipc_transporte.csv`) reconstruido a partir del
    archivo oficial del BCR, ya que el .xlsx original llegó corrupto. El CSV trae
    columnas: concepto, Anio, Mes, ipc.

    Returns: DataFrame [concepto, Anio, Mes, ipc, Periodo].
    """
    ruta = ruta or config.FUENTE_IPC
    out = pd.read_csv(ruta)
    if out.empty:
        raise ValueError("El IPC no contiene valores legibles.")
    out["Periodo"] = pd.to_datetime(dict(year=out["Anio"], month=out["Mes"], day=1))
    return out.sort_values(["concepto", "Periodo"]).reset_index(drop=True)


def cargar_produccion(ruta=None):
    """Producción anual FAOSTAT pivotada por cultivo.

    Returns: DataFrame [Anio, Grupo, area_ha, rendimiento_kg_ha, produccion_t].
    """
    ruta = ruta or config.FUENTE_PRODUCCION
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    item_a_grupo = {v: k for k, v in config.PRODUCTO_A_ITEM_PRODUCCION.items()}
    df = df[df["Item"].isin(item_a_grupo)].copy()
    df["Grupo"] = df["Item"].map(item_a_grupo)
    elem_map = {"Area harvested": "area_ha", "Yield": "rendimiento_kg_ha", "Production": "produccion_t"}
    df["metric"] = df["Element"].map(elem_map)
    df = df.dropna(subset=["metric"])
    out = (df.pivot_table(index=["Year", "Grupo"], columns="metric", values="Value", aggfunc="first")
             .reset_index().rename(columns={"Year": "Anio"}))
    out.columns.name = None
    return out.sort_values(["Grupo", "Anio"]).reset_index(drop=True)


def cargar_fuente_externa(nombre):
    """Lector genérico: 'combustible' | 'precipitacion' | 'ipc' | 'produccion'."""
    lectores = {
        "combustible": cargar_combustible,
        "precipitacion": cargar_precipitacion,
        "ipc": cargar_ipc,
        "produccion": cargar_produccion,
    }
    if nombre not in lectores:
        raise ValueError(f"Fuente desconocida: {nombre}. Use una de {list(lectores)}.")
    return lectores[nombre]()
