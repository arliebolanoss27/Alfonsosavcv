"""
Prueba la detección sobre fotos (sin cámara): dibuja las cajas y guarda el resultado en scratch/detecciones/.

Uso:
    python scripts/probar_imagen.py foto1.jpg foto2.jpg ...
    python scripts/probar_imagen.py                      # usa las 10 fotos propias reservadas para validar
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
from PIL import Image, ImageOps
import numpy as np

import config
from detector import detectar, precargar

SALIDA = config.BASE_DIR / "scratch" / "detecciones"


def cargar(ruta: Path) -> np.ndarray:
    im = ImageOps.exif_transpose(Image.open(ruta).convert("RGB"))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def main():
    rutas = [Path(a) for a in sys.argv[1:]]
    if not rutas:
        rutas = [Path(l) for l in (config.BASE_DIR / "dataset_deteccion" / "listas" / "valid_propios.txt").read_text().splitlines() if l.strip()]
    SALIDA.mkdir(parents=True, exist_ok=True)
    precargar()
    for r in rutas:
        img = cargar(r)
        dets = detectar(img, conf=config.DET_CONF_MIN)
        escala = max(img.shape[:2]) / 1000
        for d in dets:
            x0, y0, x1, y1 = d.caja
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 200, 0), max(2, int(3 * escala)))
            cv2.putText(img, f"{d.clase} {d.confianza * 100:.0f}%", (x0, max(y0 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9 * escala, (0, 255, 255), max(2, int(2 * escala)))
        destino = SALIDA / f"{r.stem}.jpg"
        cv2.imwrite(str(destino), img)
        print(f"{r.name[:50]:50} -> {[(d.clase, round(d.confianza, 2)) for d in dets] or 'sin detecciones'}")
    print(f"\nImágenes con cajas guardadas en {SALIDA}")


if __name__ == "__main__":
    main()
