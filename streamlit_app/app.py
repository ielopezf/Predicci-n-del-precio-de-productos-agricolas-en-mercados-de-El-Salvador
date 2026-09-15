"""Prototipo Streamlit — Predicción de precios agrícolas en El Salvador.

En Streamlit Community Cloud usa directamente ``api/model_registry.py`` y los
artefactos versionados en ``models/`` y ``data/processed/``. Si se define la
variable de entorno ``API_URL``, conserva el modo cliente HTTP para consumir la
API Flask desplegada por separado.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
API_DIR = ROOT_DIR / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))

API_URL = os.environ.get("API_URL", "").strip().rstrip("/")

st.set_page_config(page_title="Predicción de precios agrícolas — El Salvador", layout="wide")


# --------------------------------------------------------------------------- #
# Capa de acceso: API externa si API_URL existe; registro local en caso contrario
# --------------------------------------------------------------------------- #
@st.cache_resource
def cargar_registry():
    import model_registry as reg
    return reg


def _slugs_a_productos(reg, valor):
    if not valor:
        return None
    slugs = [s.strip() for s in valor.split(",") if s.strip()]
    desconocidos = [s for s in slugs if s not in reg.SLUG_A_PRODUCTO]
    if desconocidos:
        raise ValueError(f"Producto(s) no reconocido(s): {desconocidos}")
    return [reg.SLUG_A_PRODUCTO[s] for s in slugs]


@st.cache_data(ttl=300)
def api_get(ruta, params=None):
    """Obtiene datos desde Flask o, en Cloud, directamente desde model_registry."""
    params = params or {}

    if API_URL:
        r = requests.get(f"{API_URL}{ruta}", params=params, timeout=30)
        r.raise_for_status()
        return r.json()

    reg = cargar_registry()

    if ruta == "/api/productos":
        return reg.PRODUCTOS
    if ruta == "/api/modelos":
        return [
            {"nombre": n, "tipo": reg.TIPO_MODELO[n], "disponible": reg.MODELOS.get(n) is not None}
            for n in reg.TIPO_MODELO
        ]
    if ruta == "/api/metricas":
        return reg.metricas_modelo(params.get("modelo", "xgboost_optimizado"))
    if ruta == "/api/prediccion":
        productos = _slugs_a_productos(reg, params.get("productos"))
        df = reg.predicciones(params.get("modelo", "xgboost_optimizado"), productos=productos).copy()
        df["Fecha"] = df["Fecha"].astype(str)
        return df.to_dict(orient="records")
    if ruta == "/api/importancia":
        return reg.importancia_modelo(
            params.get("modelo", "xgboost_optimizado"), top=int(params.get("top", 15))
        )
    if ruta == "/api/equidad":
        return reg.equidad_modelo(params.get("modelo", "xgboost_optimizado"))

    raise ValueError(f"Ruta no soportada: {ruta}")


@st.cache_data(ttl=300)
def api_get_opcional(ruta, params=None):
    """Como api_get, pero devuelve None cuando un recurso opcional no está disponible."""
    try:
        if API_URL:
            r = requests.get(f"{API_URL}{ruta}", params=params, timeout=30)
            if r.status_code >= 400:
                return None
            return r.json()
        return api_get(ruta, params)
    except (KeyError, ValueError):
        return None


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
    st.error(f"No se pudo conectar con la API configurada ({API_URL}).\n\n{e}")
    st.stop()
except Exception as e:
    st.error(
        "No se pudieron cargar los datos/modelos del prototipo. Verifica que "
        "`data/processed/dataset_modelado.csv` y los archivos de `models/` estén "
        f"versionados en el repositorio.\n\nDetalle: {e}"
    )
    st.stop()

if not modelos_disp:
    st.warning("Todavía no hay modelos entrenados disponibles en `models/`.")
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
