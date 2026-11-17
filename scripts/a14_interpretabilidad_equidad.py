"""a14_interpretabilidad_equidad.py — SHAP, importancia de variables y equidad (fairness) con los modelos reentrenados.

Usa el dataset causal (a13) y los modelos de models/. El conjunto de prueba es el último 20 % (igual que a13).
Salidas en reports/etapa3/ y reports/figures/:
  importancia_variables.csv   importancia nativa + SHAP medio |valor| por modelo y variable
  importancia_por_familia.csv peso de cada familia (precio pasado, calendario, combustible, clima, IPC, producción)
  figuras shap_resumen_<modelo>.png / shap_barras_<modelo>.png / importancia_familias.png
  equidad_por_producto.csv, equidad_por_temporada.csv, equidad_brechas.csv, reporte_equidad_etapa3.txt
Uso: python scripts/a14_interpretabilidad_equidad.py [--modelos regresion_lineal xgboost xgboost_optimizado]
Requiere shap (pip install shap). Si TreeExplainer falla se usa el explicador por permutación.
"""
import argparse, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, features as ft, evaluation as ev, fairness as fa  # noqa: E402
from prediccion_precios import models_baseline as mb  # noqa: E402
warnings.filterwarnings("ignore")

FAMILIAS = {
    "precio pasado": lambda c: c.startswith(("precio_lag", "media_movil", "desv_movil")),
    "calendario": lambda c: c in ("mes", "trimestre", "semana_anio", "mes_sin", "mes_cos", "semana_sin", "semana_cos", "es_cosecha"),
    "combustible": lambda c: c in ("diesel", "gas_regular", "gas_especial"),
    "clima (lluvia)": lambda c: c in ("precipitacion_mm", "lluvia_max_mm"),
    "IPC alimentos": lambda c: c.startswith("ipc"),
    "producción": lambda c: c in ("area_ha", "produccion_t", "rendimiento_kg_ha"),
}


def familia(c):
    return next((f for f, fn in FAMILIAS.items() if fn(c)), "otras (producto/año)")


def shap_valores(modelo, X):
    import shap
    try:
        if hasattr(modelo, "coef_"):
            ex = shap.LinearExplainer(modelo, X)
            return ex(X)
        return shap.TreeExplainer(modelo)(X)
    except Exception:
        return shap.Explainer(modelo.predict, X.iloc[:100])(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modelos", nargs="+", default=["regresion_lineal", "xgboost", "xgboost_optimizado"])
    a = ap.parse_args()
    out = RAIZ / "reports" / "etapa3"; out.mkdir(parents=True, exist_ok=True)
    fig_dir = config.FIGURES_DIR; fig_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(config.DATASET_CAUSAL if config.DATASET_CAUSAL.exists() else config.DATASET_MODELADO,
                       parse_dates=[config.COL_FECHA])
    X, y, meta = ft.construir_matriz_modelado(data, return_meta=True)
    X = X.astype(float)
    _, X_te, _, y_te = ev.split_temporal(X, y)
    meta_te = meta.iloc[-len(y_te):].reset_index(drop=True)
    cosecha = data.set_index([config.COL_FECHA, config.COL_PRODUCTO])["es_cosecha"]
    es_cos = [cosecha.get((f, p), 0) for f, p in zip(meta_te[config.COL_FECHA], meta_te[config.COL_PRODUCTO])]
    prod = meta_te[config.COL_PRODUCTO].to_numpy()

    imp_filas, fam_filas, eq_prod, eq_temp, brechas = [], [], [], [], []
    modelos = {n: mb.cargar_modelo(n) for n in a.modelos}
    modelos["persistencia"] = None
    for nombre, m in modelos.items():
        p = X_te["precio_lag1"].to_numpy() if m is None else m.predict(X_te)
        tp = fa.metricas_por_grupo(y_te, p, prod).reset_index().assign(modelo=nombre)
        tt = fa.metricas_por_temporada(y_te, p, es_cos).reset_index().assign(modelo=nombre)
        eq_prod.append(tp); eq_temp.append(tt)
        brechas.append({"modelo": nombre, "brecha_productos": fa.brecha_equidad(tp.set_index("grupo")),
                        "brecha_temporada": fa.brecha_equidad(tt.set_index("grupo"))})
        if m is None:
            continue
        sv = shap_valores(m, X_te)
        vals = np.abs(sv.values).mean(axis=0)
        nat = (np.asarray(m.feature_importances_, float) if hasattr(m, "feature_importances_")
               else np.abs(np.asarray(m.coef_, float)).ravel())
        t = pd.DataFrame({"modelo": nombre, "variable": X_te.columns, "shap_medio_abs": vals,
                          "importancia_nativa": nat})
        t["familia"] = t["variable"].map(familia)
        imp_filas.append(t)
        f = t.groupby("familia")["shap_medio_abs"].sum()
        fam_filas.append((f / f.sum() * 100).rename(nombre))
        import shap
        plt.figure(); shap.summary_plot(sv, X_te, show=False, max_display=15)
        plt.title(f"SHAP — {nombre}"); plt.savefig(fig_dir / f"shap_resumen_{nombre}.png", dpi=130, bbox_inches="tight"); plt.close()
        top = t.sort_values("shap_medio_abs").tail(12)
        plt.figure(figsize=(7, 4.5)); plt.barh(top["variable"], top["shap_medio_abs"])
        plt.xlabel("SHAP medio |valor| (USD)"); plt.title(f"Variables más influyentes — {nombre}")
        plt.savefig(fig_dir / f"shap_barras_{nombre}.png", dpi=130, bbox_inches="tight"); plt.close()
        print(f"{nombre}: top-5 SHAP ->", ", ".join(t.sort_values("shap_medio_abs", ascending=False)["variable"].head(5)))

    pd.concat(imp_filas).to_csv(out / "importancia_variables.csv", index=False)
    fam = pd.DataFrame(fam_filas).round(2); fam.to_csv(out / "importancia_por_familia.csv")
    fam.T.plot(kind="barh", stacked=True, figsize=(8, 4)); plt.xlabel("% de la importancia SHAP"); plt.legend(fontsize=7)
    plt.title("Peso de cada familia de variables"); plt.savefig(fig_dir / "importancia_familias.png", dpi=130, bbox_inches="tight"); plt.close()
    pd.concat(eq_prod).round(4).to_csv(out / "equidad_por_producto.csv", index=False)
    pd.concat(eq_temp).round(4).to_csv(out / "equidad_por_temporada.csv", index=False)
    br = pd.DataFrame(brechas).round(2); br.to_csv(out / "equidad_brechas.csv", index=False)
    ref = a.modelos[0]
    t_prod = pd.concat(eq_prod); t_temp = pd.concat(eq_temp)
    txt = (fa.reporte_texto(t_prod[t_prod["modelo"] == ref].set_index("grupo"),
                            t_temp[t_temp["modelo"] == ref].set_index("grupo"))
           + "\n\nNota: el R² por producto puede ser negativo fuera de muestra cuando la varianza del producto en el "
             "periodo de prueba es casi nula (p. ej. arroz); la comparación justa es el MAPE y la razón frente a la persistencia."
           + "\nMitigación: se reporta el error por producto y el semáforo de confiabilidad en la aplicación; "
             "el pronóstico operativo es la persistencia con intervalos empíricos.")
    (out / "reporte_equidad_etapa3.txt").write_text(txt, encoding="utf-8")
    print("\nFamilias (% SHAP):\n", fam.to_string()); print("\nBrechas:\n", br.to_string(index=False))
    print("\nListo. Archivos en", out, "y", fig_dir)


if __name__ == "__main__":
    main()
