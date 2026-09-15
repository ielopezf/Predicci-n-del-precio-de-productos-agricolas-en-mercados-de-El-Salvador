"""app.py — Prototipo Streamlit de despliegue (CRISP-DM: Despliegue - Etapa 2).

Consume la API Flask (`../api/app.py`) para mostrar, por producto, el precio
real vs. el predicho por cada modelo entrenado — en modo BACKTEST sobre el
tramo de test histórico. No es un pronóstico del futuro: no existen valores
reales futuros de combustible/IPC/clima para alimentar el modelo, así que
inventar esas semanas sería engañoso (ver la sección de limitaciones al final).

Ejecutar (con la API ya corriendo en otra terminal):  streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:5000")

st.set_page_config(page_title="Predicción de precios agrícolas — El Salvador", layout="wide")


@st.cache_data(ttl=300)
def api_get(ruta, params=None):
    r = requests.get(f"{API_URL}{ruta}", params=params, timeout=30)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300)
def api_get_opcional(ruta, params=None):
    """Como api_get, pero devuelve None en vez de lanzar si el API responde 4xx.

    Usado para endpoints donde un error es un resultado válido (p. ej. un
    modelo que no expone importancia de variables), no una falla real.
    """
    r = requests.get(f"{API_URL}{ruta}", params=params, timeout=30)
    if r.status_code >= 400:
        return None
    return r.json()


st.title("Predicción de precios agrícolas — El Salvador")
st.caption(
    "Backtest histórico: precio real vs. predicho sobre el tramo de test que el modelo "
    "nunca vio al entrenarse. **No es un pronóstico del futuro** — no hay valores reales "
    "futuros de combustible, IPC o clima con los que alimentar al modelo."
)

try:
    productos_disp = api_get("/api/productos")
    modelos_disp = [m for m in api_get("/api/modelos") if m["disponible"]]
except requests.exceptions.RequestException as e:
    st.error(f"No se pudo conectar con la API ({API_URL}). ¿Está corriendo `python api/app.py`?\n\n{e}")
    st.stop()

if not modelos_disp:
    st.warning("La API está viva pero todavía no hay modelos entrenados en `models/`.")
    st.stop()

# --------------------------------------------------------------------------- #
# Barra lateral: selección de productos y modelo
# --------------------------------------------------------------------------- #
st.sidebar.header("Selección")
nombres_producto = [p["producto"] for p in productos_disp]
slug_por_producto = {p["producto"]: p["slug"] for p in productos_disp}

productos_sel = st.sidebar.multiselect("Producto(s)", nombres_producto, default=nombres_producto[:2])
nombres_modelo = [m["nombre"] for m in modelos_disp]
etiquetas_modelo = {m["nombre"]: f"{m['nombre']} ({m['tipo']})" for m in modelos_disp}
modelo_sel = st.sidebar.selectbox("Modelo", nombres_modelo, format_func=lambda n: etiquetas_modelo[n])

if not productos_sel:
    st.info("Selecciona al menos un producto en la barra lateral.")
    st.stop()

slugs_sel = ",".join(slug_por_producto[p] for p in productos_sel)

# --------------------------------------------------------------------------- #
# Métricas globales del modelo
# --------------------------------------------------------------------------- #
metricas = api_get("/api/metricas", {"modelo": modelo_sel})
st.subheader(f"Métricas del modelo — {modelo_sel}")
cols = st.columns(4)
for col, (k, v) in zip(cols, metricas["globales"].items()):
    col.metric(k, v)

with st.expander("Métricas por producto"):
    st.dataframe(pd.DataFrame(metricas["por_producto"]), width="stretch")
if metricas["por_temporada"]:
    with st.expander("Métricas cosecha vs. no cosecha"):
        st.dataframe(pd.DataFrame(metricas["por_temporada"]), width="stretch")

# --------------------------------------------------------------------------- #
# Series real vs. predicho, por producto
# --------------------------------------------------------------------------- #
st.subheader("Precio real vs. predicho (test histórico)")
pred = pd.DataFrame(api_get("/api/prediccion", {"modelo": modelo_sel, "productos": slugs_sel}))
if pred.empty:
    st.warning("Sin datos de predicción para esa combinación de producto/modelo.")
else:
    pred["Fecha"] = pd.to_datetime(pred["Fecha"])
    for producto in productos_sel:
        sub = pred[pred["Producto"] == producto].set_index("Fecha")[["precio_real", "precio_predicho"]]
        if sub.empty:
            continue
        st.markdown(f"**{producto}**")
        st.line_chart(sub)

# --------------------------------------------------------------------------- #
# Importancia de variables (si el modelo la soporta)
# --------------------------------------------------------------------------- #
importancia = api_get_opcional("/api/importancia", {"modelo": modelo_sel, "top": 15})
if isinstance(importancia, list):
    st.subheader("Importancia de variables")
    df_imp = pd.DataFrame(importancia).set_index("feature")
    st.bar_chart(df_imp["importancia"])

# --------------------------------------------------------------------------- #
# Evaluación ética y de sesgos
# --------------------------------------------------------------------------- #
st.subheader("Evaluación ética y de sesgos")
equidad = api_get("/api/equidad", {"modelo": modelo_sel})
st.dataframe(pd.DataFrame(equidad["por_producto"]), width="stretch")
if equidad["reporte"]:
    with st.expander("Limitaciones y consideraciones éticas", expanded=False):
        st.text(equidad["reporte"])
