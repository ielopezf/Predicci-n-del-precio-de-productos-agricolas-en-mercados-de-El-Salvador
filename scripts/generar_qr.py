"""generar_qr.py — Genera el código QR de la app publicada (para portada, diapositiva y README).

Uso: python scripts/generar_qr.py [URL] [salida.png]
Requiere: pip install "qrcode[pil]"
"""
import sys
from pathlib import Path
import qrcode

URL = sys.argv[1] if len(sys.argv) > 1 else "https://prediccionpreciosagricolassv.streamlit.app/"
SALIDA = Path(sys.argv[2] if len(sys.argv) > 2 else Path(__file__).resolve().parents[1] / "reports" / "figures" / "qr_app.png")
SALIDA.parent.mkdir(parents=True, exist_ok=True)
qrcode.make(URL, box_size=12, border=4).save(SALIDA)
print(f"QR de {URL} guardado en {SALIDA}")
