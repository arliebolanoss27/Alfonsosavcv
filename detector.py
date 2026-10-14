"""
Inferencia de detección: todos los objetos del cuadro, cada uno con su caja.
    detectar(frame) -> list[Deteccion]
Usa modelos/best_det.pt (entrenado con scripts/entrenar_deteccion.py).
"""

from dataclasses import dataclass

import numpy as np

import config

_modelo = None


@dataclass
class Deteccion:
    clase: str
    confianza: float
    caja: tuple[int, int, int, int]  # x0, y0, x1, y1 en píxeles del frame original


def _cargar_modelo():
    global _modelo
    if _modelo is None:
        from ultralytics import YOLO
        ruta = config.MODELOS_DIR / config.MODELO_DET_PESOS
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró {ruta}. Entrena con scripts/entrenar_deteccion.py")
        _modelo = YOLO(str(ruta))
    return _modelo


def detectar(frame: np.ndarray, conf: float = config.DET_CONF_MIN) -> list[Deteccion]:
    r = _cargar_modelo().predict(source=frame, imgsz=config.DET_IMGSZ, conf=conf, verbose=False)[0]
    salida = []
    for caja, c, k in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy()):
        x0, y0, x1, y1 = (int(v) for v in caja)
        salida.append(Deteccion(r.names[int(k)], float(c), (x0, y0, x1, y1)))
    return sorted(salida, key=lambda d: d.confianza, reverse=True)
