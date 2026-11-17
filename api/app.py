"""app.py — API Flask de despliegue (CRISP-DM: Despliegue - Etapa 2).

Despliegue: ver render.yaml (Render, gunicorn). Endpoints: /api/health, /api/productos, /api/modelos,
/api/prediccion, /api/metricas, /api/importancia, /api/equidad y /api/pronostico.

Sirve, en modo BACKTEST (precio real vs. predicho sobre el tramo de test que
ningún modelo vio durante el entrenamiento/ajuste), los modelos entrenados en
los notebooks 03 (baselines) y 04 (avanzados/ensemble). No ofrece pronóstico
hacia el futuro: no hay valores reales futuros de combustible/IPC/clima, así
que "predecir" aquí significa evaluar la calidad del modelo sobre historia
real, no inventar semanas que no han ocurrido (ver `fairness.LIMITACIONES_ETICAS`).

Ejecutar:  python app.py   (o `flask --app app run`), desde esta carpeta.
"""

from __future__ import annotations

from flask import Flask, jsonify, request
from flask_cors import CORS

import model_registry as reg

app = Flask(__name__)
CORS(app)


def _slugs_a_productos(valor):
    """'maiz_blanco,frijol_rojo_de_seda_nacional' -> nombres reales de producto."""
    if not valor:
        return None
    slugs = [s.strip() for s in valor.split(",") if s.strip()]
    productos, desconocidos = [], []
    for s in slugs:
        if s in reg.SLUG_A_PRODUCTO:
            productos.append(reg.SLUG_A_PRODUCTO[s])
        else:
            desconocidos.append(s)
    if desconocidos:
        raise ValueError(f"Producto(s) no reconocido(s): {desconocidos}")
    return productos


@app.errorhandler(KeyError)
def _modelo_no_encontrado(e):
    return jsonify({"error": str(e).strip('"\'')}), 404


@app.errorhandler(ValueError)
def _valor_invalido(e):
    return jsonify({"error": str(e)}), 400


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "modelos_disponibles": reg.MODELOS_DISPONIBLES})


@app.get("/api/productos")
def productos():
    return jsonify(reg.PRODUCTOS)


@app.get("/api/modelos")
def modelos():
    return jsonify([
        {"nombre": n, "tipo": reg.TIPO_MODELO[n], "disponible": reg.MODELOS.get(n) is not None}
        for n in reg.TIPO_MODELO
    ])


@app.get("/api/prediccion")
def prediccion():
    modelo = request.args.get("modelo", "xgboost_optimizado")
    productos_sel = _slugs_a_productos(request.args.get("productos"))
    df = reg.predicciones(modelo, productos=productos_sel)
    df = df.copy()
    df["Fecha"] = df["Fecha"].astype(str)
    return jsonify(df.to_dict(orient="records"))


@app.get("/api/metricas")
def metricas():
    modelo = request.args.get("modelo", "xgboost_optimizado")
    return jsonify(reg.metricas_modelo(modelo))


@app.get("/api/importancia")
def importancia():
    modelo = request.args.get("modelo", "xgboost_optimizado")
    top = int(request.args.get("top", 15))
    tabla = reg.importancia_modelo(modelo, top=top)
    if tabla is None:
        return jsonify({"error": f"'{modelo}' no tiene importancia de variables disponible (la LSTM usa solo precios pasados)."}), 400
    return jsonify(tabla)


@app.get("/api/equidad")
def equidad():
    modelo = request.args.get("modelo", "xgboost_optimizado")
    return jsonify(reg.equidad_modelo(modelo))


@app.get("/api/pronostico")
def pronostico():
    """Pronóstico de la semana siguiente: persistencia (referencia) con intervalos empíricos, más la predicción de la
    regresión lineal y de la LSTM (si está disponible). ?producto=TOMATE (opcional)."""
    from prediccion_precios import pronostico as pr
    import pandas as pd
    hist = pr.cargar_historia()
    grupo = request.args.get("producto", "").upper() or None
    try:
        precio = float(request.args["precio"]) if "precio" in request.args else None
        fb = pd.Timestamp(request.args["fecha_base"]) if "fecha_base" in request.args else None
        if precio is not None and grupo is None:
            return jsonify({"error": "'precio' requiere indicar 'producto'."}), 400
        if grupo:
            tabla = pr.pronosticar(hist, grupo, precio, fb).tabla()
        else:
            tabla = pr.pronosticar_todos(hist)
    except (ValueError, KeyError) as e:
        return jsonify({"error": str(e)}), 400
    if not request.args.get("precio") and "fecha_base" not in request.args:
        alt = reg.pronostico_siguiente()
        campos = ("valor", "lim_inf_80", "lim_sup_80", "lim_inf_90", "lim_sup_90", "mape_prueba_%", "semaforo")
        for nombre in ("regresion_lineal", "lstm"):
            for c in campos:
                col = f"pronostico_{nombre}" if c == "valor" else f"{c}_{nombre}"
                tabla[col] = tabla["grupo"].map(lambda g: (alt.get(g, {}).get(nombre) or {}).get(c))
        tabla = tabla.astype(object).where(tabla.notna(), None)
    tabla["fecha_base"] = tabla["fecha_base"].astype(str)
    tabla["fecha_objetivo"] = tabla["fecha_objetivo"].astype(str)
    return jsonify(tabla.to_dict(orient="records"))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
