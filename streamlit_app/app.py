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

import altair as alt
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


NOMBRES_VARIABLES = {
    "precio_lag1": "Precio hace 1 semana", "precio_lag2": "Precio hace 2 semanas",
    "precio_lag4": "Precio hace 4 semanas", "precio_lag8": "Precio hace 8 semanas",
    "media_movil_4": "Promedio 4 semanas", "media_movil_8": "Promedio 8 semanas", "media_movil_12": "Promedio 12 semanas",
    "desv_movil_4": "Variación 4 semanas", "desv_movil_8": "Variación 8 semanas", "desv_movil_12": "Variación 12 semanas",
    "mes": "Mes", "Mes": "Mes", "Anio": "Año", "trimestre": "Trimestre", "semana_anio": "Semana del año",
    "mes_sin": "Ciclo anual (mes, seno)", "mes_cos": "Ciclo anual (mes, coseno)",
    "semana_sin": "Ciclo anual (semana, seno)", "semana_cos": "Ciclo anual (semana, coseno)",
    "es_cosecha": "Temporada de cosecha", "diesel": "Precio del diésel", "gas_regular": "Gasolina regular",
    "gas_especial": "Gasolina especial", "precipitacion_mm": "Lluvia semanal (mm)", "lluvia_max_mm": "Lluvia máxima (mm)",
    "ipc_alimentos": "IPC de alimentos", "area_ha": "Superficie cultivada", "produccion_t": "Producción anual",
    "rendimiento_kg_ha": "Rendimiento por hectárea",
}

NOMBRES_MODELOS = {
    "regresion_lineal": "Regresión lineal", "random_forest": "Random Forest", "xgboost": "XGBoost",
    "xgboost_baseline": "XGBoost", "xgboost_optimizado": "XGBoost optimizado", "ensemble_stacking": "Ensamble (stacking)",
    "red_neuronal": "Red neuronal", "lstm": "LSTM", "lstm (tramo propio)": "LSTM (tramo propio)",
    "persistencia (último precio)": "Persistencia (último precio)",
}
TIPOS_MODELO = {"baseline": "modelo base", "avanzado": "modelo avanzado"}

NOMBRES_METRICAS = {
    "MAE": ("MAE (USD)", "Error absoluto medio: cuántos dólares se equivoca el modelo en promedio."),
    "RMSE": ("RMSE (USD)", "Raíz del error cuadrático medio: como el MAE, pero penaliza más los errores grandes."),
    "MAPE_%": ("MAPE (%)", "Error porcentual absoluto medio: cuánto se equivoca, en %, respecto al precio real."),
    "R2": ("R²", "Coeficiente de determinación: qué parte de la variación del precio explica el modelo (puede ser negativo fuera de muestra)."),
}

COLUMNAS_PRONOSTICO = {
    "producto": "Producto", "horizonte_sem": "Semanas a pronosticar", "fecha_base": "Fecha del último precio",
    "fecha_objetivo": "Fecha pronosticada", "precio_base": "Último precio (USD)", "pronostico": "Persistencia, referencia: pronóstico (USD)",
    "lim_inf_80": "Límite inferior 80 % (USD)", "lim_sup_80": "Límite superior 80 % (USD)",
    "lim_inf_90": "Límite inferior 90 % (USD)", "lim_sup_90": "Límite superior 90 % (USD)",
    "mape_prueba_%": "Error de prueba (MAPE, %)", "semaforo": "Confiabilidad", "avisos": "Avisos",
}
MOSTRAR_CARGA_ARCHIVO = False   # carga de precios vigentes oculta; poner True para habilitarla
MODELOS_ALTERNATIVOS = ("regresion_lineal", "lstm")
CONFIABILIDAD = {"verde": "Alta", "amarillo": "Media", "rojo": "Baja", "gris": "No evaluada"}


# Unidad de venta al mayoreo de cada producto en los informes del MAG: (unidad corta, descripción).
UNIDADES_VENTA = {
    "ARROZ": ("quintal", "quintal (100 lb)"), "FRIJOL": ("quintal", "quintal (100 lb)"),
    "MAIZ": ("quintal", "quintal (100 lb)"), "NARANJA": ("ciento", "ciento (100 unidades)"),
    "TOMATE": ("caja", "caja (275–300 unidades)"),
}
TEXTO_UNIDADES = ("Precios de mayoreo del MAG, en USD por la unidad de venta de cada producto: arroz, frijol y maíz por "
                  "quintal (100 lb); naranja por ciento (100 unidades); tomate por caja (275–300 unidades).")


def _unidad(producto, larga=False):
    """Unidad de venta según el nombre del producto (acepta 'ARROZ' o 'ARROZ ORO PRIMERA CLASE IMPORTADO')."""
    n = str(producto).upper().replace("Í", "I")
    for clave, (corta, desc) in UNIDADES_VENTA.items():
        if clave in n:
            return desc if larga else corta
    return "unidad" if not larga else "unidad de venta"


def _quitar_antiguedad(txt):
    """Quita de los avisos el de antigüedad del dato (esa nota vive en el bloque de carga de archivo)."""
    return " | ".join(a for a in str(txt).split(" | ") if a and a != "nan" and "antigüedad" not in a)


def _nombre_variable(c):
    return NOMBRES_VARIABLES.get(c, str(c).replace("_", " ").capitalize())


def _nombre_modelo(n):
    return NOMBRES_MODELOS.get(n, str(n).replace("_", " ").capitalize())


COLOR_NIVEL = {"Bajo (<5 %)": "#2e9e5b", "Medio (5–15 %)": "#e0a526", "Alto (>15 %)": "#d64545"}


def _nivel_mape(v):
    return "Bajo (<5 %)" if v < 5 else ("Medio (5–15 %)" if v <= 15 else "Alto (>15 %)")


def _barras(df, x, y, titulo, fmt=".2f", color=None, altura=260, etiqueta_x="Producto"):
    """Barras con el valor sobre cada barra (sin tabla)."""
    enc_color = color if color is not None else alt.value("#4c78a8")
    base = alt.Chart(df).encode(x=alt.X(f"{x}:N", sort=None, title=None, axis=alt.Axis(labelAngle=0, labelLimit=120)),
                                y=alt.Y(f"{y}:Q", title=titulo))
    barras = base.mark_bar().encode(color=enc_color, tooltip=[alt.Tooltip(x, title=etiqueta_x), alt.Tooltip(y, format=fmt, title=titulo)])
    texto = base.mark_text(dy=-7, fontSize=12).encode(text=alt.Text(f"{y}:Q", format=fmt))
    return (barras + texto).properties(height=altura)


def _color_nivel():
    return alt.Color("nivel:N", scale=alt.Scale(domain=list(COLOR_NIVEL), range=list(COLOR_NIVEL.values())),
                     legend=alt.Legend(title="Error (MAPE)", orient="bottom"))


def _reflujo_vinetas(texto):
    """Une las líneas cortadas a mano dentro de cada viñeta ('- ...') para que fluyan como párrafo."""
    items = []
    for linea in texto.splitlines():
        s = linea.strip()
        if not s:
            continue
        if s.startswith("- "):
            items.append(s[2:])
        elif items:
            items[-1] += " " + s
    return "\n".join(f"- {i}" for i in items)


def _grafico_pronostico(hist_g, tabla_g, unidad="unidad", semanas_hist=26, modelos=None):
    """Precio observado (últimas semanas del MAG) + pronóstico con bandas de 80 % y 90 %: muestra de dónde parte la predicción."""
    obs = hist_g.sort_values("Fecha").tail(semanas_hist)[["Fecha", "Precio"]].rename(columns={"Precio": "valor"})
    obs["serie"] = "Precio observado"
    f = tabla_g.sort_values("horizonte_sem").copy()
    f["Fecha"] = pd.to_datetime(f["fecha_objetivo"])
    base = pd.DataFrame([{"Fecha": pd.to_datetime(f["fecha_base"].iloc[0]), "pronostico": f["precio_base"].iloc[0],
                          "lim_inf_80": f["precio_base"].iloc[0], "lim_sup_80": f["precio_base"].iloc[0],
                          "lim_inf_90": f["precio_base"].iloc[0], "lim_sup_90": f["precio_base"].iloc[0]}])
    f = pd.concat([base, f[["Fecha", "pronostico", "lim_inf_80", "lim_sup_80", "lim_inf_90", "lim_sup_90"]]],
                  ignore_index=True)
    pron = f[["Fecha", "pronostico"]].rename(columns={"pronostico": "valor"})
    pron["serie"] = "Pronóstico"
    lineas = pd.concat([obs, pron], ignore_index=True)
    x = alt.X("Fecha:T", title=None)
    y_esc = alt.Scale(zero=False)
    dom_b = ["Intervalo 80 %", "Intervalo 90 %"]
    color_b = alt.Color("banda:N", scale=alt.Scale(domain=dom_b, range=["#4c78a8", "#a9c4de"]),
                        legend=alt.Legend(orient="bottom", title=None, symbolType="square"))
    b90 = alt.Chart(f.assign(banda="Intervalo 90 %")).mark_area(opacity=0.35).encode(
        x=x, y=alt.Y("lim_inf_90:Q", scale=y_esc, title=f"Precio (USD por {unidad})"), y2="lim_sup_90:Q", color=color_b)
    b80 = alt.Chart(f.assign(banda="Intervalo 80 %")).mark_area(opacity=0.45).encode(
        x=x, y=alt.Y("lim_inf_80:Q", scale=y_esc), y2="lim_sup_80:Q", color=color_b)
    ln = alt.Chart(lineas).mark_line(strokeWidth=2).encode(
        x=x, y=alt.Y("valor:Q", scale=y_esc),
        color=alt.Color("serie:N", scale=alt.Scale(domain=["Precio observado", "Pronóstico"], range=["#6b7280", "#2e9e5b"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.StrokeDash("serie:N", scale=alt.Scale(domain=["Precio observado", "Pronóstico"], range=[[1, 0], [6, 4]]),
                                  legend=None),
        tooltip=["serie", alt.Tooltip("Fecha:T"), alt.Tooltip("valor:Q", format=".2f", title="USD")])
    pto = alt.Chart(f.head(1)).mark_point(size=90, filled=True, color="#2e9e5b").encode(x=x, y=alt.Y("pronostico:Q", scale=y_esc))
    capas = b90 + b80 + ln + pto
    mods_ok = {n: d for n, d in (modelos or {}).items() if d}
    if mods_ok:
        fecha_obj = pd.to_datetime(tabla_g["fecha_objetivo"].iloc[0])
        pm = pd.DataFrame([{"Fecha": fecha_obj, "valor": d["valor"], "modelo": _nombre_modelo(n),
                            "lo": d.get("lim_inf_80", d["valor"]), "hi": d.get("lim_sup_80", d["valor"])}
                           for n, d in mods_ok.items()])
        dom = list(pm["modelo"])
        rng = [{"Regresión lineal": "#d97706", "LSTM": "#8b5cf6"}.get(d, "#444") for d in dom]
        col_m = alt.Color("modelo:N", scale=alt.Scale(domain=dom, range=rng),
                          legend=alt.Legend(orient="bottom", title=None))
        capas = capas + alt.Chart(pm).mark_rule(strokeWidth=3, opacity=0.6).encode(
            x=x, y=alt.Y("lo:Q", scale=y_esc), y2="hi:Q", color=col_m) \
            + alt.Chart(pm).mark_point(size=110, filled=True, shape="diamond").encode(
            x=x, y=alt.Y("valor:Q", scale=y_esc), color=col_m,
            tooltip=["modelo", alt.Tooltip("valor:Q", format=".2f", title="USD"),
                     alt.Tooltip("lo:Q", format=".2f", title="Rango 80 %, inferior"),
                     alt.Tooltip("hi:Q", format=".2f", title="Rango 80 %, superior")])
    return capas.resolve_scale(color="independent").properties(height=280)


def _pastel(df, cat, val, umbral=1.0):
    """Pastel con nombre y porcentaje fuera de las porciones ≥ umbral %; la leyenda lista todas."""
    d = df.sort_values(val, ascending=False).reset_index(drop=True)
    d["pct"] = d[val] / d[val].sum() * 100
    d["etiqueta"] = [f"{c} {p:.0f} %" if p >= umbral else "" for c, p in zip(d[cat], d["pct"])]
    base = alt.Chart(d).encode(
        theta=alt.Theta(f"{val}:Q", stack=True),
        order=alt.Order(f"{val}:Q", sort="descending"),
        color=alt.Color(f"{cat}:N", sort=list(d[cat]), legend=alt.Legend(orient="right", title=None, symbolLimit=0)),
        tooltip=[cat, alt.Tooltip("pct:Q", format=".1f", title="% del total")])
    arco = base.mark_arc(outerRadius=120, stroke="white", strokeWidth=1)
    texto = base.mark_text(radius=160, size=12).encode(text="etiqueta:N")
    return (arco + texto).properties(height=400)


def _grafico_comparacion():
    """MAPE de todos los modelos (tabla del reentrenamiento causal) con la persistencia resaltada."""
    ruta = ROOT_DIR / "reports" / "etapa3" / "reentrenamiento_tabla.csv"
    if not ruta.exists():
        return None
    d = pd.read_csv(ruta)
    d = d.rename(columns={d.columns[0]: "Modelo"})[["Modelo", "MAPE_%"]].sort_values("MAPE_%")
    d["tipo"] = ["Método operativo" if m.startswith("persistencia") else "Modelo de ML" for m in d["Modelo"]]
    d["Modelo"] = d["Modelo"].map(_nombre_modelo)
    base = alt.Chart(d).encode(y=alt.Y("Modelo:N", sort=None, title=None),
                               x=alt.X("MAPE_%:Q", title="Error porcentual medio (MAPE, %) en prueba — menor es mejor"))
    barras = base.mark_bar().encode(
        color=alt.Color("tipo:N", scale=alt.Scale(domain=["Método operativo", "Modelo de ML"],
                                                  range=["#2e9e5b", "#4c78a8"]), legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Modelo", alt.Tooltip("MAPE_%", format=".2f", title="MAPE (%)")])
    texto = base.mark_text(align="left", dx=4).encode(text=alt.Text("MAPE_%:Q", format=".2f"))
    return (barras + texto).properties(height=300)


st.title("Predicción de precios agrícolas — El Salvador")


def vista_pronostico():
    sys.path.insert(0, str(ROOT_DIR / "src"))
    from prediccion_precios import pronostico as pr

    st.caption(
        "Pronóstico del precio de la **semana siguiente**. La **persistencia** (el último precio) es la referencia, con "
        "intervalos empíricos: ningún modelo de ML la superó de forma robusta en la prueba. Como alternativas se muestran "
        "la **regresión lineal** y la **LSTM**, que suponen que combustible, lluvia, IPC y producción se mantienen en su "
        "último valor conocido. Los intervalos son orientativos: en la prueba cubrieron 63 % (nominal 80 %) y 73 % (nominal 90 %)."
    )

    st.caption(TEXTO_UNIDADES)

    @st.cache_data
    def _historia():
        return pr.cargar_historia()

    hist = _historia()
    ult_fecha = hist["Fecha"].max()

    grupos = sorted(hist["Grupo"].unique())
    precios, fecha_base, archivo = {}, None, None
    if MOSTRAR_CARGA_ARCHIVO:
        with st.expander("Cargar precios vigentes (archivo)", expanded=True):
            st.info(
                f"Los datos históricos terminan el {ult_fecha.date()}. Para obtener el pronóstico a la fecha actual, "
                "cargue aquí el precio vigente; sin archivo, el pronóstico corresponde a esa última fecha."
            )
            st.markdown(
                "**Formato requerido** (.csv o .xlsx, máx. 2 MB): exactamente tres columnas "
                "`Fecha` (AAAA-MM-DD), `Producto` (el nombre debe contener ARROZ, FRIJOL, MAÍZ, NARANJA o TOMATE) "
                "y `Precio` (USD por la unidad de venta del producto, número con punto decimal: quintal para arroz, frijol y maíz; "
                "ciento para naranja; caja para tomate). Use los precios de la plaza Calle Gerardo Barrios (granos) y del "
                "Mercado Mayorista La Tiendona (hortalizas y frutas), que son los de la serie histórica; no los de las tablas "
                "nacionales o departamentales. Se usa la última fila de cada producto. "
                "Los precios muy fuera del rango histórico se rechazan."
            )
            st.code(pr.plantilla_csv(hist), language="csv")
            st.download_button("Descargar plantilla", pr.plantilla_csv(hist).encode("utf-8"),
                               "plantilla_precios.csv", "text/csv")
            archivo = st.file_uploader("Cargar archivo", type=["csv", "xlsx"])
            if archivo is not None and archivo.size > 2 * 1024 * 1024:
                st.error("El archivo supera 2 MB.")
                archivo = None
        if archivo is not None:
            try:
                ult = pr.leer_reporte_mag(archivo, hist)
                precios.update({g: p for g, (f, p) in ult.items()})
                fecha_base = max(f for f, _ in ult.values()) if ult else None
                st.success(f"Reporte leído: {len(ult)} producto(s) reconocido(s).")
            except Exception as e:
                st.error(f"No se pudo leer el reporte: {e}")

    sel = st.multiselect("Productos a consultar", grupos, default=grupos)
    h_sel = 1
    if not sel:
        st.info("Seleccione al menos un producto.")
        return
    tabla = pr.pronosticar_todos(hist, precios, fecha_base)
    tabla = tabla[tabla["grupo"].isin(sel)]
    nivel = {"verde": ("green", "Confiabilidad alta (error de prueba < 5 %)"),
             "amarillo": ("orange", "Confiabilidad media (error de prueba 5–15 %)"),
             "rojo": ("red", "Confiabilidad baja (error de prueba > 15 %)"),
             "gris": ("gray", "Confiabilidad no evaluada")}
    try:   # alternativas de ML; solo coherentes con la serie histórica (no con un archivo cargado)
        alt_ml = {} if precios else cargar_registry().pronostico_siguiente()
    except Exception:
        alt_ml = {}
    disponibles = [n for n in MODELOS_ALTERNATIVOS if any(v.get(n) is not None for v in alt_ml.values())]
    r = tabla[tabla["horizonte_sem"] == h_sel]
    filas = []
    for _, fila in r.iterrows():
        g = fila["grupo"]
        base = {"Producto": fila["producto"].title(), "Unidad de venta": _unidad(g, larga=True),
                "Fecha pronosticada": str(fila["fecha_objetivo"]), "Último precio (USD)": f"{fila['precio_base']:.2f}"}
        filas.append({**base, "Método": "Persistencia (referencia)", "Pronóstico (USD)": f"{fila['pronostico']:.2f}",
                      "Rango 80 %": f"{fila['lim_inf_80']:.2f} – {fila['lim_sup_80']:.2f}",
                      "Rango 90 %": f"{fila['lim_inf_90']:.2f} – {fila['lim_sup_90']:.2f}",
                      "Confiabilidad": CONFIABILIDAD.get(fila["semaforo"], "—")})
        for n in disponibles:
            d = alt_ml.get(g, {}).get(n)
            if not d:
                continue
            filas.append({**base, "Método": _nombre_modelo(n), "Pronóstico (USD)": f"{d['valor']:.2f}",
                          "Rango 80 %": f"{d['lim_inf_80']:.2f} – {d['lim_sup_80']:.2f}" if "lim_inf_80" in d else "—",
                          "Rango 90 %": f"{d['lim_inf_90']:.2f} – {d['lim_sup_90']:.2f}" if "lim_inf_90" in d else "—",
                          "Confiabilidad": CONFIABILIDAD.get(d.get("semaforo"), "—")})
    resumen = pd.DataFrame(filas)
    st.subheader("Resumen del pronóstico de la semana siguiente")
    st.dataframe(resumen, hide_index=True, width="stretch")
    st.caption(f"Datos históricos del MAG hasta el {ult_fecha.date()}. Los precios están en USD por la unidad de venta indicada. "
               "Cada método tiene su propio rango y su propia confiabilidad: la persistencia, con la variación histórica del precio; "
               "la regresión lineal y la LSTM, con sus errores en el tramo de prueba.")
    if len(disponibles) < len(MODELOS_ALTERNATIVOS):
        faltan = ", ".join(_nombre_modelo(n) for n in MODELOS_ALTERNATIVOS if n not in disponibles)
        st.caption(f"No disponible en este momento: {faltan}.")
    for g in sel:
        f = tabla[(tabla["grupo"] == g) & (tabla["horizonte_sem"] == h_sel)].iloc[0]
        st.subheader(f["producto"].title())
        u = _unidad(g)
        mods = alt_ml.get(g, {})
        cols = st.columns(1 + len(disponibles))
        cols[0].metric(f"Persistencia, referencia ({f['fecha_objetivo']})", f"{f['pronostico']:.2f} USD por {u}")
        c_r, e_r = nivel[f["semaforo"]]
        cols[0].caption(f"Rango 80 %: {f['lim_inf_80']:.2f} – {f['lim_sup_80']:.2f}\n\nRango 90 %: "
                        f"{f['lim_inf_90']:.2f} – {f['lim_sup_90']:.2f}")
        cols[0].markdown(f":{c_r}-badge[{e_r}]")
        for col, n in zip(cols[1:], disponibles):
            d = mods.get(n)
            if not d:
                col.metric(_nombre_modelo(n), "No disponible")
                continue
            col.metric(_nombre_modelo(n), f"{d['valor']:.2f} USD por {u}",
                       delta=f"{d['valor'] - f['pronostico']:+.2f} frente a la referencia", delta_color="off")
            if "lim_inf_80" in d:
                col.caption(f"Rango 80 %: {d['lim_inf_80']:.2f} – {d['lim_sup_80']:.2f}\n\nRango 90 %: "
                            f"{d['lim_inf_90']:.2f} – {d['lim_sup_90']:.2f}")
                c_m, e_m = nivel.get(d["semaforo"], ("gray", "Sin dato"))
                col.markdown(f":{c_m}-badge[{e_m}]")
        st.write(pr.explicacion(f).replace("$", "\\$"))
        hist_g = hist[hist["Grupo"] == g]
        if g in precios:
            origen = f"último precio del archivo cargado ({f['fecha_base']})"
        else:
            origen = f"último precio registrado por el MAG ({hist_g['Fecha'].max().date()})"
        st.altair_chart(_grafico_pronostico(hist_g, tabla[tabla["grupo"] == g], u,
                                                  modelos={n: mods.get(n) for n in disponibles}), width="stretch")
        st.caption(f"Datos usados: serie semanal de precios del MAG (línea gris, últimas 26 semanas). "
                   f"El pronóstico parte del {origen}, por **{f['precio_base']:.2f} USD por {u}**; "
                   "las bandas azules son el rango donde se espera el precio: oscura 80 %, clara 90 % (orientativas; en la prueba cubrieron 63 % y 73 %). "
                   "Los rombos son lo que estiman la regresión lineal y la LSTM, cada uno con su barra de rango del 80 % calculada con sus propios errores de prueba; "
                   "son una alternativa a la referencia, no un pronóstico mejor.".replace("$", "\\$"))
        for a in _quitar_antiguedad(f["avisos"]).split(" | "):
            if a:
                st.warning(a)
    for n in disponibles:
        nm = _nombre_modelo(n)
        for campo, rot in (("valor", "Pronóstico (USD)"), ("lim_inf_80", "Rango 80 % inferior (USD)"),
                           ("lim_sup_80", "Rango 80 % superior (USD)"), ("lim_inf_90", "Rango 90 % inferior (USD)"),
                           ("lim_sup_90", "Rango 90 % superior (USD)"), ("mape_prueba_%", "Error de prueba (MAPE, %)"),
                           ("semaforo", "Confiabilidad")):
            vals = tabla["grupo"].map(lambda g, n=n, campo=campo: (alt_ml.get(g, {}).get(n) or {}).get(campo))
            tabla[f"{nm} — {rot}"] = vals.map(CONFIABILIDAD) if campo == "semaforo" else vals
    descarga = tabla.assign(semaforo=lambda d: d["semaforo"].map(CONFIABILIDAD),
                            avisos=lambda d: d["avisos"].map(_quitar_antiguedad),
                            unidad=lambda d: d["grupo"].map(lambda g: _unidad(g, larga=True))) \
        .drop(columns=["grupo"]).rename(columns={**COLUMNAS_PRONOSTICO, "unidad": "Unidad de venta"})
    st.download_button("Descargar tabla del pronóstico (CSV)", descarga.to_csv(index=False).encode("utf-8-sig"),
                       "pronostico.csv", "text/csv")
    st.caption("Confiabilidad según el error de prueba: alta < 5 %, media 5–15 % (meta del proyecto), baja > 15 %. "
               "Describe el desempeño pasado, no garantiza el futuro.")


def vista_backtest():
    st.caption(TEXTO_UNIDADES)
    st.caption(
        "Backtest histórico (modelos reentrenados con el tratamiento causal de datos: sin interpolar ni recortar picos): "
    "precio real vs. predicho sobre el tramo de test que el modelo "
        "nunca vio al entrenarse. **No es un pronóstico del futuro** — no hay valores reales "
        "futuros de combustible, IPC o clima con los que alimentar al modelo."
    )

    try:
        productos_disp = api_get("/api/productos")
        modelos_disp = [m for m in api_get("/api/modelos") if m["disponible"]]
    except requests.exceptions.RequestException as e:
        st.error(f"No se pudo conectar con la API configurada ({API_URL}).\n\n{e}")
        return
    except Exception as e:
        st.error(
            "No se pudieron cargar los datos/modelos del prototipo. Verifica que "
            "`data/processed/dataset_modelado.csv` y los archivos de `models/` estén "
            f"versionados en el repositorio.\n\nDetalle: {e}"
        )
        return

    if not modelos_disp:
        st.warning("Todavía no hay modelos entrenados disponibles en `models/`.")
        return

    # --------------------------------------------------------------------------- #
    # Comparación de todos los modelos
    # --------------------------------------------------------------------------- #
    st.subheader("Comparación de modelos")
    g = _grafico_comparacion()
    if g is not None:
        st.altair_chart(g, width="stretch")
        st.caption("La persistencia (repetir el último precio) es el método operativo y la referencia que ningún modelo "
                   "supera. La LSTM se evalúa en un tramo propio de secuencias: no es comparable fila a fila.")

    # --------------------------------------------------------------------------- #
    # Barra lateral: selección de productos y modelo
    # --------------------------------------------------------------------------- #
    st.subheader("Selección")
    col_p, col_m = st.columns([2, 1])
    nombres_producto = [p["producto"] for p in productos_disp]
    slug_por_producto = {p["producto"]: p["slug"] for p in productos_disp}

    productos_sel = col_p.multiselect("Productos", nombres_producto, default=nombres_producto[:2])
    nombres_modelo = [m["nombre"] for m in modelos_disp]
    etiquetas_modelo = {m["nombre"]: f"{_nombre_modelo(m['nombre'])} ({TIPOS_MODELO.get(m['tipo'], m['tipo'])})" for m in modelos_disp}
    modelo_sel = col_m.selectbox("Modelo", nombres_modelo, format_func=lambda n: etiquetas_modelo[n])

    if not productos_sel:
        st.info("Selecciona al menos un producto.")
        return

    slugs_sel = ",".join(slug_por_producto[p] for p in productos_sel)

    # --------------------------------------------------------------------------- #
    # Métricas globales y por producto
    # --------------------------------------------------------------------------- #
    metricas = api_get("/api/metricas", {"modelo": modelo_sel})
    st.subheader(f"Métricas del modelo — {_nombre_modelo(modelo_sel)}")
    cols = st.columns(4)
    for col, (k, v) in zip(cols, metricas["globales"].items()):
        etiqueta, ayuda = NOMBRES_METRICAS.get(k, (k, None))
        col.metric(etiqueta, v, help=ayuda)

    st.markdown("**Error por producto**")
    pp = pd.DataFrame(metricas["por_producto"])
    pp["nivel"] = pp["MAPE_%"].map(_nivel_mape)
    c1, c2, c3 = st.columns(3)
    c1.altair_chart(_barras(pp, "grupo", "MAPE_%", "Error porcentual medio (MAPE, %)", color=_color_nivel()), width="stretch")
    c2.altair_chart(_barras(pp, "grupo", "MAE", "Error absoluto medio (MAE, USD)"), width="stretch")
    c3.altair_chart(_barras(pp, "grupo", "R2", "Coeficiente de determinación (R²)", fmt=".2f"), width="stretch")
    st.caption("El MAE está en USD por la unidad de venta de cada producto, así que no es comparable entre productos. "
               "El MAPE es la métrica comparable entre productos. El R² puede ser negativo fuera de muestra cuando el "
               "precio casi no varía en el periodo de prueba (p. ej. arroz); en ese caso un MAPE bajo es lo que importa.")

    if metricas["por_temporada"]:
        st.markdown("**Cosecha vs. no cosecha**")
        pt = pd.DataFrame(metricas["por_temporada"])
        pt["grupo"] = pt["grupo"].replace({"cosecha": "Cosecha", "no_cosecha": "No cosecha"})
        t1, t2 = st.columns(2)
        t1.altair_chart(_barras(pt, "grupo", "MAPE_%", "Error porcentual medio (MAPE, %)", altura=220, etiqueta_x="Temporada"), width="stretch")
        t2.altair_chart(_barras(pt, "grupo", "MAE", "Error absoluto medio (MAE, USD)", altura=220, etiqueta_x="Temporada"), width="stretch")

    # --------------------------------------------------------------------------- #
    # Importancia de variables (por permutación, comparable entre modelos)
    # --------------------------------------------------------------------------- #
    st.subheader("Importancia de variables")
    importancia = api_get_opcional("/api/importancia", {"modelo": modelo_sel, "top": 100})
    if isinstance(importancia, list) and importancia:
        df_imp = pd.DataFrame(importancia)
        df_imp["importancia"] = df_imp["importancia"].clip(lower=0)   # una importancia negativa es ruido: cuenta como 0
        total = df_imp["importancia"].sum()
        if total > 0:
            top = df_imp.sort_values("importancia", ascending=False).head(15)
            datos = top.assign(grupo=top["feature"].map(_nombre_variable))[["grupo", "importancia"]]
            resto = total - top["importancia"].sum()
            if resto > 1e-12:   # el pastel suma 100 %: el resto de variables va en una porción aparte
                datos = pd.concat([datos, pd.DataFrame([{"grupo": "Otras variables", "importancia": resto}])],
                                  ignore_index=True)
            st.altair_chart(_pastel(datos, "grupo", "importancia", umbral=4), width="stretch")
            fam = (df_imp.groupby("familia")["importancia"].sum() / total * 100)
            pasado, prod = float(fam.get("Precio pasado", 0.0)), float(fam.get("Producción", 0.0))
            mostrar_prod = prod >= 10
            # enteros que suman exactamente 100 (método del mayor resto)
            partes = [pasado, prod if mostrar_prod else 0.0]
            partes.append(max(100.0 - sum(partes), 0.0))
            base = [int(x) for x in partes]
            for i in sorted(range(3), key=lambda i: partes[i] - base[i], reverse=True)[:100 - sum(base)]:
                base[i] += 1
            pasado_i, prod_i, otras_i = base
            texto = (f"Los datos más relevantes son los **precios pasados** ({pasado_i} % de la importancia): "
                     "el modelo se apoya casi por completo en lo que ya ocurrió con el precio. ")
            if mostrar_prod:
                texto += (f"La **producción** también pesa ({prod_i} %), pero al ser una cifra anual constante por producto "
                          "actúa como identificador del producto y el año más que como causa del precio. ")
            texto += (f"El resto de variables (calendario, combustible, lluvia, IPC de alimentos) aportan en conjunto "
                      f"{otras_i} %: no mejoran de forma apreciable la predicción del precio futuro.")
            st.markdown(texto)
            st.caption("Importancia por permutación: cuánto empeora el error al barajar la variable en el tramo de prueba. "
                       "El pastel muestra las 15 variables principales y agrupa el resto en \"Otras variables\"; los porcentajes suman 100 %.")
        else:
            st.info("Las variables de este modelo no cambian su error al barajarse: no hay importancia que mostrar.")
    else:
        st.info("La LSTM usa solo una ventana de precios pasados (sin variables externas), por lo que no tiene "
                "importancia de variables que mostrar.")

    # --------------------------------------------------------------------------- #
    # Evaluación ética y de sesgos
    # --------------------------------------------------------------------------- #
    st.subheader("Evaluación ética y de sesgos")
    equidad = api_get("/api/equidad", {"modelo": modelo_sel})
    mp = [r["MAPE_%"] for r in equidad["por_producto"]]
    e1, e2 = st.columns(2)
    if mp and min(mp) > 0:
        e1.metric("Brecha entre productos (peor/mejor MAPE)", f"{max(mp) / min(mp):.1f}x",
                  help="Refleja que el tomate es mucho más volátil que el arroz; no es un sesgo del modelo.")
    if equidad.get("por_temporada"):
        mt = [r["MAPE_%"] for r in equidad["por_temporada"]]
        if min(mt) > 0:
            e2.metric("Brecha cosecha vs. no cosecha", f"{max(mt) / min(mt):.1f}x")
    st.caption("Comunicar el error de cada producto y no solo el promedio. Los productos con MAPE alto "
               "aparecen en rojo en las gráficas de arriba.")
    if equidad["reporte"]:
        st.markdown("**Limitaciones y consideraciones éticas**")
        st.markdown(_reflujo_vinetas(equidad["reporte"].split("Limitaciones y consideraciones éticas:")[-1]))

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
            sub = (pred[pred["Producto"] == producto].set_index("Fecha")[["precio_real", "precio_predicho"]]
                   .rename(columns={"precio_real": "Precio real", "precio_predicho": "Precio predicho"}))
            if sub.empty:
                continue
            st.markdown(f"**{producto}** — USD por {_unidad(producto)}")
            st.line_chart(sub)


tab_back, tab_pron = st.tabs(["Backtest y comparación de modelos", "Pronóstico (1 semana)"])
with tab_back:
    vista_backtest()
with tab_pron:
    vista_pronostico()
