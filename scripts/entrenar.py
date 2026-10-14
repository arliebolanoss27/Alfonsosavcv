"""
Script de entrenamiento para YOLO26s-cls en el reto ToolLighthouse.
Entrena el clasificador con transfer learning sobre las 9 clases de componentes.
Ajustes respetados:
- imgsz = 224
- epochs = 10
- batch = 32 (apropiado para 8GB VRAM de RTX 5060)
- hsv_h = 0.0 (sin alteraciones de tono de color para proteger bandas de resistencias y LEDs)
- device = 0 (GPU dedicada)
"""

import sys
from pathlib import Path

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from ultralytics import YOLO
import torch
import config


def entrenar_modelo():
    print("=" * 60)
    print("INICIANDO ENTRENAMIENTO YOLO26s-cls (ToolLighthouse)")
    print("=" * 60)
    
    # Verificación de aceleración GPU
    cuda_ok = torch.cuda.is_available()
    device = 0 if cuda_ok else "cpu"
    print(f"Dispositivo de cómputo: {torch.cuda.get_device_name(0) if cuda_ok else 'CPU'} (device={device})")

    # Cargar pesos preentrenados de YOLO26s-cls
    modelo_weights = config.MODELO_DEFAULT_WEIGHTS
    print(f"Cargando arquitectura base: {modelo_weights}")
    model = YOLO(modelo_weights)

    # Ruta absoluta del dataset
    dataset_path = str(config.DATASET_PROCESADO_DIR.resolve())
    print(f"Ruta del dataset: {dataset_path}")

    # Ejecutar entrenamiento
    results = model.train(
        data=dataset_path,
        epochs=10,
        imgsz=config.MODELO_IMGSZ,
        batch=32,
        device=device,
        workers=4,
        hsv_h=0.0,      # Restricción obligatoria: no alterar matiz de color
        hsv_s=0.2,      # Pequeña variación de saturación
        hsv_v=0.3,      # Variación de brillo / exposición
        degrees=15.0,   # Rotación ligera
        fliplr=0.5,     # Volteo horizontal
        flipud=0.0,     # Sin volteo vertical agresivo
        project="runs/classify",
        name="tool_lighthouse_yolo26s",
        exist_ok=True,
        verbose=True,
    )

    print("\nEntrenamiento completado exitosamente.")
    return model, results


if __name__ == "__main__":
    entrenar_modelo()
