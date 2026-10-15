"""
Prueba SOLO el asistente de voz, sin cámara ni detección.

Uso:  python scripts/probar_voz.py

Pulsa Enter, habla 4 segundos ("¿dónde está la resistencia?") y mira qué entendió y qué responde.
Escribe  t  para probar con texto escrito (sin micrófono) y  q  para salir.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
import voz
from enlace_serial import EnlaceSerial


def guardar_wav(audio):
    """Guarda la última grabación en scratch/ultima_grabacion.wav para poder escucharla."""
    import wave
    import numpy as np
    carpeta = config.BASE_DIR / "scratch"
    carpeta.mkdir(exist_ok=True)
    with wave.open(str(carpeta / "ultima_grabacion.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(voz.FRECUENCIA)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())


def mostrar(texto: str, enlace: EnlaceSerial):
    clase = voz.interpretar(texto)
    respuesta = voz.respuesta_para(clase)
    print(f'  Oí:        "{texto}"')
    print(f"  Entendí:   {clase}")
    print(f"  Responde:  {respuesta}")
    if clase == "apagar":
        enlace.apagar_laser()
    elif clase:
        enlace.apuntar(clase)
    voz.hablar(respuesta)


def main():
    enlace = EnlaceSerial(simulado=True)
    escucha = voz.Escucha()
    print("Cargando modelo de voz (la primera vez descarga ~460 MB)...")
    escucha.calentar()
    print("Listo.\n")
    while True:
        op = input("Enter = hablar | t = escribir texto | q = salir > ").strip().lower()
        if op == "q":
            break
        if op == "t":
            mostrar(input("  Escribe la frase: "), enlace)
        else:
            print(f"  Habla ahora ({config.VOZ_SEGUNDOS:.0f} s)...")
            audio = escucha.grabar()
            guardar_wav(audio)
            nivel = escucha.ultimo_pico
            print(f"  Nivel de audio: {nivel:.2f}" + ("  <- MUY BAJO: acércate al micrófono o sube su volumen" if nivel < 0.05 else ""))
            print("  Procesando...")
            mostrar(escucha.transcribir(audio), enlace)
        print()


if __name__ == "__main__":
    main()
