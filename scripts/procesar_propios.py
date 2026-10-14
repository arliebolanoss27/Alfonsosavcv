"""
Script para incorporar y procesar las fotografías y videos reales capturados por el equipo.
Soporta formatos JPG, JPEG, PNG, HEIC y videos (MP4, MOV, AVI).
Conserva intactas las fotos originales y las convierte a JPG estándar RGB en la carpeta intermedia.
Permite agrupar por archivo/sesión para evitar fugas de datos durante el split.
"""

import sys
from pathlib import Path
import hashlib

# Añadir el directorio raíz al path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from PIL import Image
import cv2
from tqdm import tqdm

# Registro de soporte HEIC mediante pillow_heif
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import config

DATOS_PROPIOS_CRUDOS = config.BASE_DIR / "datos_propios" / "crudos"
OUTPUT_DIR = config.BASE_DIR / "dataset_intermedio" / "propios"

EXTENSIONES_FOTO = {".jpg", ".jpeg", ".png", ".heic", ".webp"}
EXTENSIONES_VIDEO = {".mp4", ".mov", ".avi"}

# Frames a saltar en videos (extraer 1 frame cada FPS aprox, ej. 25 frames)
FRAME_INTERVAL_VIDEO = 25


def inicializar_carpetas_fotos_reales():
    """Crea la estructura de carpetas para que el usuario coloque sus fotografías."""
    DATOS_PROPIOS_CRUDOS.mkdir(parents=True, exist_ok=True)
    for clase in config.CLASES:
        clase_dir = DATOS_PROPIOS_CRUDOS / clase
        clase_dir.mkdir(parents=True, exist_ok=True)
        # Crear un archivo .gitkeep o explicativo si está vacío
        gitkeep = clase_dir / ".keep"
        if not gitkeep.exists():
            gitkeep.touch()
    print(f"[Datos Propios] Estructura creada en: {DATOS_PROPIOS_CRUDOS}")


def procesar_imagen_propia(file_path: Path, dest_dir: Path, clase: str) -> bool:
    try:
        with Image.open(file_path) as img:
            img_rgb = img.convert("RGB")
            # Identificador único por hash de archivo
            with open(file_path, "rb") as f:
                fhash = hashlib.md5(f.read(32768)).hexdigest()[:8]
            dest_file = dest_dir / f"propio_{clase}_{file_path.stem}_{fhash}.jpg"
            img_rgb.save(dest_file, "JPEG", quality=95)
            return True
    except Exception as e:
        print(f"Error procesando imagen {file_path.name}: {e}")
        return False


def procesar_video_propio(file_path: Path, dest_dir: Path, clase: str) -> int:
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        print(f"Error abriendo video {file_path.name}")
        return 0

    frame_count = 0
    saved_count = 0
    vid_stem = file_path.stem

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % FRAME_INTERVAL_VIDEO == 0:
            # Convertir BGR a RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            dest_file = dest_dir / f"propio_{clase}_{vid_stem}_f{frame_count:05d}.jpg"
            img.save(dest_file, "JPEG", quality=95)
            saved_count += 1
        frame_count += 1

    cap.release()
    return saved_count


def procesar_fotos_y_videos_propios():
    inicializar_carpetas_fotos_reales()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    conteo = {c: 0 for c in config.CLASES}
    archivos_encontrados = 0

    for clase in config.CLASES:
        clase_origen = DATOS_PROPIOS_CRUDOS / clase
        clase_destino = OUTPUT_DIR / clase
        clase_destino.mkdir(parents=True, exist_ok=True)

        archivos = [
            f for f in clase_origen.iterdir()
            if f.is_file() and f.name != ".keep"
        ]

        if not archivos:
            continue

        archivos_encontrados += len(archivos)
        print(f"\nProcesando fotos/videos propios para '{clase}' ({len(archivos)} archivos)...")

        for fpath in tqdm(archivos, desc=f"Procesando {clase}"):
            ext = fpath.suffix.lower()
            if ext in EXTENSIONES_FOTO:
                if procesar_imagen_propia(fpath, clase_destino, clase):
                    conteo[clase] += 1
            elif ext in EXTENSIONES_VIDEO:
                frames_extraidos = procesar_video_propio(fpath, clase_destino, clase)
                conteo[clase] += frames_extraidos

    print("\n--- Resumen de procesamiento de fotos/videos propios ---")
    if archivos_encontrados == 0:
        print("Aún no se han colocado archivos en 'datos_propios/crudos/<clase>/'.")
    else:
        for cls_name in config.CLASES:
            cant = conteo.get(cls_name, 0)
            if cant > 0:
                print(f"  - {cls_name}: {cant} imágenes extraídas")


if __name__ == "__main__":
    procesar_fotos_y_videos_propios()
