"""
Busca las cámaras que Windows ve (índices 0 a 5) y guarda una foto de cada una
en scratch/camara_<indice>.jpg para identificar cuál es el iPhone 16.

Antes: conectar el iPhone por USB y abrir DroidCam o Iriun en el teléfono y en la PC.
Uso:  python scripts/listar_camaras.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2

import config

SALIDA = config.BASE_DIR / "scratch"


def main():
    SALIDA.mkdir(exist_ok=True)
    encontradas = 0
    for i in range(6):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMARA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMARA_HEIGHT)
        for _ in range(15):  # calentamiento: los primeros frames salen rotos o negros
            ok, frame = cap.read()
        cap.release()
        if not ok:
            print(f"Índice {i}: abre pero no entrega imagen")
            continue
        ruta = SALIDA / f"camara_{i}.jpg"
        cv2.imwrite(str(ruta), frame)
        h, w = frame.shape[:2]
        print(f"Índice {i}: {w}x{h} -> {ruta}")
        encontradas += 1
    if encontradas == 0:
        print("No se encontró ninguna cámara. Revisa el cable USB y que Iriun/DroidCam esté abierto en ambos lados.")
    else:
        print("\nAbre las fotos y mira cuál es el iPhone. Ese número es el que va en --camara (o CAMARA_INDEX en config.py).")


if __name__ == "__main__":
    main()
