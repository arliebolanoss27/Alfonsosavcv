"""
Entrena YOLO26 de DETECCIÓN (una caja + clase por objeto) con dataset_deteccion/.

Antes: python scripts/preparar_deteccion.py
Uso:   python scripts/entrenar_deteccion.py [--epocas 60] [--modelo yolo26s.pt] [--nombre det_yolo26s]

Al terminar mide por separado con SOLO las fotos propias (lo que importa para la demo) y copia
el mejor peso a modelos/best_det.pt.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import torch
from ultralytics import YOLO

import config

DET = config.BASE_DIR / "dataset_deteccion"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--modelo", default="yolo26s.pt")
    ap.add_argument("--nombre", default="det_yolo26s")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=960, help="resolución de entrenamiento (960 ayuda con piezas pequeñas)")
    args = ap.parse_args()

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Dispositivo: {torch.cuda.get_device_name(0) if device == 0 else 'CPU'}")

    model = YOLO(args.modelo)
    model.train(
        data=str(DET / "data.yaml"),
        epochs=args.epocas,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=4,
        hsv_h=0.0,       # sin cambiar el tono: el color identifica resistencias, LEDs, servos...
        hsv_s=0.4,
        hsv_v=0.4,
        degrees=20.0,    # las fotos propias están tomadas desde arriba: cualquier rotación es válida
        flipud=0.5,
        fliplr=0.5,
        scale=0.6,
        mosaic=1.0,      # junta varias imágenes: enseña a detectar varios objetos a la vez
        project=str(config.BASE_DIR / "runs" / "detect"),
        name=args.nombre,
        exist_ok=True,
        patience=20,
    )

    mejor = Path(model.trainer.best)
    config.MODELOS_DIR.mkdir(exist_ok=True)
    shutil.copy(mejor, config.MODELOS_DIR / "best_det.pt")
    print(f"Mejor peso copiado a {config.MODELOS_DIR / 'best_det.pt'}")

    print("\n=== Validación SOLO con fotos propias (27 fotos) ===")
    YOLO(str(mejor)).val(data=str(DET / "data_propios.yaml"), imgsz=args.imgsz, device=device, plots=True,
                         project=str(config.BASE_DIR / "runs" / "detect"), name=args.nombre + "_propias", exist_ok=True)


if __name__ == "__main__":
    main()
