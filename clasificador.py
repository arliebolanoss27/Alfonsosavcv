"""
Módulo de inferencia para ToolLighthouse.
Expone la interfaz estándar documentada:
    predecir(frame) -> tuple[str, float]
Compatible con YOLO26s-cls y cualquier variante YOLO26-cls.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import numpy as np
import config

_modelo = None


def _cargar_modelo():
    global _modelo
    if _modelo is None:
        from ultralytics import YOLO
        best_pt = config.MODELOS_DIR / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"No se encontró best.pt en {best_pt}")
        _modelo = YOLO(str(best_pt))
    return _modelo


def predecir(frame: np.ndarray) -> tuple[str, float]:
    """
    Clasifica un frame BGR capturado con OpenCV.

    Args:
        frame: imagen numpy BGR de cualquier resolución (se redimensiona internamente a 224×224).

    Returns:
        (clase, confianza): nombre de la clase ganadora (str) y
        probabilidad normalizada (float en [0, 1]).
    """
    model = _cargar_modelo()
    results = model.predict(
        source=frame,
        imgsz=config.MODELO_IMGSZ,
        verbose=False
    )
    r = results[0]
    probs = r.probs
    top1_idx = int(probs.top1)
    confianza = float(probs.top1conf)
    clase = r.names[top1_idx]
    return clase, confianza


if __name__ == "__main__":
    # Prueba de sanidad rápida
    import cv2
    cap = cv2.VideoCapture(config.CAMARA_INDEX)
    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {config.CAMARA_INDEX}. Usando imagen de prueba.")
        cap.release()
    else:
        ret, frame = cap.read()
        cap.release()
        if ret:
            clase, conf = predecir(frame)
            print(f"Predicción: {clase} ({conf*100:.1f}% confianza)")
        else:
            print("No se pudo leer frame de la cámara.")
