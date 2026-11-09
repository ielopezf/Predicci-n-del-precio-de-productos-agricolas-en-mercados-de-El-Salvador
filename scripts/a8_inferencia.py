"""a8_inferencia.py — Prueba de Diebold-Mariano e intervalos de predicción (A8).

1) DM: ¿el error de cada modelo difiere del de la persistencia? Pérdida = error absoluto,
   escalado por el error medio de la persistencia de cada producto (peso igual por producto);
   se promedia entre productos por semana -> una serie semanal. Varianza de largo plazo con
   Newey-West (rezagos h-1) y corrección de Harvey-Leybourne-Newbold; p-valor t de Student.
2) Intervalos empíricos para la persistencia (el pronóstico base): cuantiles de la variación
   relativa y/p_t - 1 por producto y horizonte, calculados SOLO con semanas previas a la prueba;
   se reporta la cobertura real en la prueba y el ancho medio.
Uso: python scripts/a8_inferencia.py --precios precios_consolidados.xlsx
"""
import argparse, sys, importlib.util
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config, pipeline_causal as pc  # noqa: E402
spec = importlib.util.spec_from_file_location("a6", RAIZ / "scripts" / "a6_afinacion.py")
a6 = importlib.util.module_from_spec(spec); spec.loader.exec_module(a6)
a4 = a6.a4
ET = RAIZ / "reports" / "etapa3"


def dm(d, h):
    d = np.asarray(d, float); n = len(d); m = d.mean(); dc = d - m
    lrv = np.sum(dc * dc) / n
    for k in range(1, h):
        lrv += 2 * (1 - k / h) * np.sum(dc[k:] * dc[:-k]) / n
    if lrv <= 0: return np.nan, np.nan
    est = m / np.sqrt(lrv / n)
    adj = np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    est *= adj
    return est, 2 * (1 - stats.t.cdf(abs(est), df=n - 1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--precios", required=True); a = ap.parse_args()
    det = pd.concat([pd.read_csv(ET / "afinado_detalle.csv"), pd.read_csv(ET / "ensamble_detalle.csv")], ignore_index=True)
    det = det[~det.modelo.isin(["persistencia", "comite_simple"])]
    det["fecha"] = pd.to_datetime(det["fecha"])
    filas = []
    for h in a4.HORIZ:
        x = det[det.horizonte == h]
        esc = x.groupby("producto")["err_pers"].mean()
        for mod, g in x.groupby("modelo"):
            g = g.assign(dif=(g["err"] - g["err_pers"]) / g["producto"].map(esc))
            serie = g.groupby("fecha")["dif"].mean().sort_index()
            est, p = dm(serie.to_numpy(), h)
            filas.append({"horizonte": h, "modelo": mod, "semanas": len(serie), "dif_media_escalada": serie.mean(),
                          "DM": est, "p_valor": p})
    res = pd.DataFrame(filas); res.to_csv(ET / "diebold_mariano.csv", index=False)
    print("== DM vs persistencia (dif<0 = el modelo es mejor) =="); print(res.round(3).to_string(index=False))

    base = pd.read_csv(config.DATASET_MODELADO, parse_dates=[config.COL_FECHA])
    df = a4.features_origen(pc.con_externas(pc.semanal_causal(a.precios), base))
    out = []
    for h in a4.HORIZ:
        d = a6.preparar(df, h, [])
        d["rel"] = d["y"] / d["p0"] - 1
        cal = d[d["ok"] & (d["fecha_obj"] < a4.INI_TEST)]; te = d[d["ok"] & (d["fecha_obj"] >= a4.INI_TEST)]
        for nivel in (0.8, 0.9):
            a_ = (1 - nivel) / 2
            for g, c in cal.groupby("Grupo"):
                lo, hi = c["rel"].quantile([a_, 1 - a_])
                t = te[te.Grupo == g]
                dentro = ((t["rel"] >= lo) & (t["rel"] <= hi)).mean()
                out.append({"horizonte": h, "producto": g, "nivel_nominal": nivel, "cobertura_prueba": dentro,
                            "ancho_relativo_medio_%": (hi - lo) * 100, "n_prueba": len(t)})
    cov = pd.DataFrame(out); cov.to_csv(ET / "intervalos_persistencia.csv", index=False)
    for nivel in (0.8, 0.9):
        print(f"\n== Cobertura en la prueba, intervalo nominal {int(nivel*100)}% ==")
        print(cov[cov.nivel_nominal == nivel].pivot(index="producto", columns="horizonte", values="cobertura_prueba").round(2))
    print("\n== Ancho relativo (%) del intervalo 80% =="); print(cov[cov.nivel_nominal == .8].pivot(index="producto", columns="horizonte", values="ancho_relativo_medio_%").round(1))
    print("\ncobertura media:", cov.groupby("nivel_nominal").cobertura_prueba.mean().round(3).to_dict())


if __name__ == "__main__":
    main()
