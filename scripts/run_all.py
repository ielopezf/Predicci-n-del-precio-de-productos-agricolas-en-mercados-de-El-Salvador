"""run_all.py — Ejecuta en orden todos los análisis de la etapa final y deja los resultados en reports/etapa3/.

Uso básico (desde la raíz del repositorio, con el entorno virtual activo):
    python scripts/run_all.py

Opciones:
    --precios RUTA           Excel de precios (por defecto data/interim/precios_consolidados.xlsx)
    --con-lstm               Incluye el LSTM (requiere tensorflow o keras; es lento)
    --omitir a2,a10          Omite pasos por su clave (ver PASOS)
    --solo a1,a4             Ejecuta solo esos pasos
    --sin-pruebas            No corre pytest al inicio
Los pasos se ejecutan en el orden de dependencias: a6 antes de a7, a8 y a10.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))
from prediccion_precios import config  # noqa: E402

# clave, script, descripción, ¿necesita --precios?
PASOS = [
    ("a1", "a1_baselines.py", "Referencias ingenuas y ETS/ARIMA", False),
    ("a2", "a2_auditoria.py", "Auditoría de calidad de datos y cobertura de fuentes", None),
    ("a9", "a9_sensibilidad_nulos.py", "Sensibilidad al tratamiento de nulos y valores atípicos", True),
    ("a4", "a4_horizontes.py", "Horizontes 1/2/4/8 semanas con origen móvil", True),
    ("a6", "a6_afinacion.py", "Afinación equivalente y tabla por producto", True),
    ("a7", "a7_ensamble.py", "Ensamble contraído hacia la persistencia", True),
    ("a8", "a8_inferencia.py", "Diebold-Mariano e intervalos de predicción", True),
    ("a10", "a10_prueba_lluvia_mensual.py", "Prueba de la lluvia mensual como variable", None),
    ("a11", "a11_lstm.py", "LSTM con el mismo protocolo (opcional)", True),
]


def argumentos(clave, precios, lluvia):
    if clave == "a2":
        a = ["--precios", str(precios)]
        for opt, ruta in (("--combustibles", config.FUENTE_COMBUSTIBLE),
                          ("--precip-diaria", config.FUENTE_PRECIPITACION),
                          ("--faostat", config.FUENTE_PRODUCCION)):
            if Path(ruta).exists():
                a += [opt, str(ruta)]
        if lluvia.exists():
            a += ["--lluvia-mensual", str(lluvia)]
        return a
    if clave == "a10":
        return ["--precios", str(precios), "--lluvia-mensual", str(lluvia)]
    if clave == "a1":
        return []
    return ["--precios", str(precios)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--precios", default=str(config.PRECIOS_CONSOLIDADOS))
    ap.add_argument("--lluvia-mensual", default=str(config.EXTERNAL_DIR / "lluvia_mensual_sv.xlsx"))
    ap.add_argument("--con-lstm", action="store_true")
    ap.add_argument("--omitir", default="")
    ap.add_argument("--solo", default="")
    ap.add_argument("--sin-pruebas", action="store_true")
    a = ap.parse_args()

    precios, lluvia = Path(a.precios), Path(a.lluvia_mensual)
    if not precios.exists():
        sys.exit(f"No existe el archivo de precios: {precios}\nUsa --precios RUTA.")
    omitir = {x.strip() for x in a.omitir.split(",") if x.strip()}
    solo = {x.strip() for x in a.solo.split(",") if x.strip()}
    if not lluvia.exists():
        print(f"Aviso: no se encontró {lluvia}; se omite la prueba a10 y la fila de lluvia mensual en a2.")
        omitir.add("a10")
    if not a.con_lstm:
        omitir.add("a11")
    (RAIZ / "reports" / "etapa3").mkdir(parents=True, exist_ok=True)

    trabajos = []
    if not a.sin_pruebas and not solo:
        trabajos.append(("pruebas", [sys.executable, "-m", "pytest", "-q", "tests/test_baselines.py"], "Pruebas automáticas"))
    for clave, script, desc, _ in PASOS:
        if clave in omitir or (solo and clave not in solo):
            continue
        trabajos.append((clave, [sys.executable, str(RAIZ / "scripts" / script)] + argumentos(clave, precios, lluvia), desc))

    resumen = []
    for clave, cmd, desc in trabajos:
        print(f"\n{'=' * 78}\n[{clave}] {desc}\n{'=' * 78}", flush=True)
        t0 = time.time()
        r = subprocess.run(cmd, cwd=RAIZ)
        seg = time.time() - t0
        resumen.append((clave, desc, "OK" if r.returncode == 0 else f"ERROR ({r.returncode})", seg))
        if r.returncode != 0:
            print(f"\nFalló el paso [{clave}]. Corrige el error y vuelve a ejecutar; "
                  f"puedes reanudar con --solo y las claves que faltan.")
            break
    print("\n" + "-" * 78)
    for clave, desc, est, seg in resumen:
        print(f"{clave:8s} {est:10s} {seg:7.1f} s  {desc}")
    print("-" * 78 + f"\nResultados en: {RAIZ / 'reports' / 'etapa3'}")
    sys.exit(0 if all(e == "OK" for _, _, e, _ in resumen) else 1)


if __name__ == "__main__":
    main()
