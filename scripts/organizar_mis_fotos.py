"""
Script para organizar las fotos de mis_fotos hacia datos_propios/crudos
a partir de un diccionario de asignación confirmado.
Copia los archivos sin alterar mis_fotos/.
"""

import shutil
from pathlib import Path
import config

ORIGEN_DIR = config.BASE_DIR / "mis_fotos"
DESTINO_DIR = config.DATOS_PROPIOS_DIR / "crudos"


def organizar_fotos(mapeo_confirmado: dict[str, str]):
    DESTINO_DIR.mkdir(parents=True, exist_ok=True)
    conteo = {c: 0 for c in config.CLASES}

    for archivo_nombre, clase in mapeo_confirmado.items():
        if clase not in config.CLASES:
            print(f"Clase desconocida: {clase} para {archivo_nombre}")
            continue

        src = ORIGEN_DIR / archivo_nombre
        if not src.exists():
            print(f"Archivo no encontrado: {src}")
            continue

        clase_dir = DESTINO_DIR / clase
        clase_dir.mkdir(parents=True, exist_ok=True)

        dst = clase_dir / archivo_nombre
        shutil.copy2(src, dst)
        conteo[clase] += 1

    print("\n--- Fotografías organizadas en datos_propios/crudos/ ---")
    for c in config.CLASES:
        if conteo[c] > 0:
            print(f"  - {c}: {conteo[c]} fotos")


if __name__ == "__main__":
    # Se ejecutará con el mapeo que confirme el usuario
    pass
