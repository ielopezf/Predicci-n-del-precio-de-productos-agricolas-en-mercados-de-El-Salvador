"""a7_ensamble.py — Ensamble de contracción hacia la persistencia (A7).

pronóstico = p_t * (1 + w_producto * r_comité), con r_comité = promedio de las variaciones que
predicen Ridge, Random Forest y Gradient Boosting (hiperparámetros de a6). w ∈ {0, .25, .5, .75, 1}
se elige POR PRODUCTO y horizonte con los mismos 3 pliegues de validación previos a la prueba
(w=0 es la persistencia). Se evalúa en la prueba con reentreno cada 8 semanas.
Uso: python scripts/a7_ensamble.py --precios precios_consolidados.xlsx
"""
import argparse, ast, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, pipeline_causal as pc  # noqa: E402
spec = importlib.util.spec_from_file_location("a6", RAIZ / "scripts" / "a6_afinacion.py")
a6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a6)
a4 = a6.a4
MIEMBROS = ("Ridge_var", "RF_var", "HGB_var"); GRILLA = (0, .25, .5, .75, 1.0)


def mejores_params(h):
    cv = pd.read_csv(RAIZ / "reports" / "etapa3" / "afinacion_cv.csv")
    out = {}
    for m in MIEMBROS:
        s = cv[(cv.horizonte == h) & (cv.modelo == m)].sort_values("puntaje_cv").iloc[0]
        out[m] = ast.literal_eval(s["params"])
    return out


def retorno_comite(tr, te, params, feats_m):
    rs = []
    for m in MIEMBROS:
        pred = a6.ajustar_predecir(m, params[m], tr, te, feats_m)
        rs.append(pred / te["p0"].to_numpy() - 1)
    return np.mean(rs, axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--precios", required=True); a = ap.parse_args()
    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    df = a4.features_origen(pc.con_externas(pc.semanal_causal(a.precios), base))
    feats = a4.lista_features(df)
    dums = pd.get_dummies(df["Grupo"], prefix="g", dtype=float)
    df = pd.concat([df, dums], axis=1); feats_m = feats + list(dums.columns)
    INI = a4.INI_TEST
    filas, pesos = [], []
    for h in a4.HORIZ:
        d = a6.preparar(df, h, feats_m); P = mejores_params(h)
        previa = d[d["ok"] & (d["fecha_obj"] < INI)]
        fechas = np.array(sorted(previa["fecha_obj"].unique()))
        cortes = [fechas[int(len(fechas) * q)] for q in (0.55, 0.70, 0.85)] + [fechas[-1] + pd.Timedelta(days=1)]
        cv = []
        for i in range(3):
            tr = previa[previa["fecha_obj"] < cortes[i]]
            te = previa[(previa["fecha_obj"] >= cortes[i]) & (previa["fecha_obj"] < cortes[i + 1])]
            r = retorno_comite(tr, te, P, feats_m)
            cv.append(pd.DataFrame({"g": te["Grupo"].to_numpy(), "y": te["y"].to_numpy(),
                                    "p0": te["p0"].to_numpy(), "r": r}))
        cv = pd.concat(cv)
        w_prod = {}
        for g, x in cv.groupby("g"):
            mae = {w: np.abs(x["y"] - x["p0"] * (1 + w * x["r"])).mean() for w in GRILLA}
            w_prod[g] = min(mae, key=mae.get)       # empate -> el menor w (más cerca de persistencia)
            pesos.append({"horizonte": h, "producto": g, "w": w_prod[g]})
        test = d[d["ok"] & (d["fecha_obj"] >= INI)]
        bloques = np.array_split(sorted(test["fecha_obj"].unique()), int(np.ceil(32 / a4.REFIT)))
        for bl in bloques:
            tr = d[d["ok"] & (d["fecha_obj"] < bl[0])]; te = test[test["fecha_obj"].isin(bl)]
            if len(te) == 0: continue
            r = retorno_comite(tr, te, P, feats_m)
            w = te["Grupo"].map(w_prod).to_numpy()
            filas.append(pd.DataFrame({"horizonte": h, "modelo": "ensamble_contraido", "producto": te["Grupo"].to_numpy(),
                                       "fecha": te["fecha_obj"].to_numpy(), "real": te["y"].to_numpy(),
                                       "predicho": te["p0"].to_numpy() * (1 + w * r), "persistencia": te["p0"].to_numpy()}))
            filas.append(pd.DataFrame({"horizonte": h, "modelo": "comite_simple", "producto": te["Grupo"].to_numpy(),
                                       "fecha": te["fecha_obj"].to_numpy(), "real": te["y"].to_numpy(),
                                       "predicho": te["p0"].to_numpy() * (1 + r), "persistencia": te["p0"].to_numpy()}))
    det = pd.concat(filas, ignore_index=True)
    det["err"] = (det["real"] - det["predicho"]).abs(); det["ape"] = det["err"] / det["real"] * 100
    det["err_pers"] = (det["real"] - det["persistencia"]).abs()
    det.to_csv(RAIZ / "reports" / "etapa3" / "ensamble_detalle.csv", index=False)
    pd.DataFrame(pesos).to_csv(RAIZ / "reports" / "etapa3" / "ensamble_pesos.csv", index=False)
    t = (det.groupby(["horizonte", "producto", "modelo"]).agg(MAE=("err", "mean"), MAPE=("ape", "mean"),
         MAE_pers=("err_pers", "mean")).reset_index())
    t["razon_vs_persistencia"] = t["MAE"] / t["MAE_pers"]
    t.round(4).to_csv(RAIZ / "reports" / "etapa3" / "ensamble_resumen.csv", index=False)
    print("== pesos w elegidos (0 = persistencia) =="); print(pd.DataFrame(pesos).pivot(index="producto", columns="horizonte", values="w"))
    for mod in ("ensamble_contraido", "comite_simple"):
        print(f"\n== razón MAE vs persistencia: {mod} ==")
        print(t[t.modelo == mod].pivot(index="producto", columns="horizonte", values="razon_vs_persistencia").round(3))
    print("\n== MAPE % ensamble_contraido =="); print(t[t.modelo == "ensamble_contraido"].pivot(index="producto", columns="horizonte", values="MAPE").round(2))
    print("== MAPE % persistencia =="); pp = det.drop_duplicates(["horizonte", "producto", "fecha"]).assign(ape=lambda x: (x.real - x.persistencia).abs() / x.real * 100)
    print(pp.groupby(["producto", "horizonte"]).ape.mean().unstack().round(2))


if __name__ == "__main__":
    main()
