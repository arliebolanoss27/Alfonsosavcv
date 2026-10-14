"""
ToolLighthouse con DETECCIÓN: ve todo el cuadro, dibuja una caja por objeto y no hace nada si no hay nada.

Uso:
    python main_det.py                    # cámara de config.CAMARA_INDEX, serial simulado si no hay Nano
    python main_det.py --camara 1 --puerto COM5
    python main_det.py --video prueba.mp4 # probar con un video en vez de cámara

Teclas: q o Esc = salir | espacio = pausar/reanudar el envío al Nano | r = reiniciar estabilidad
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np

import config
from detector import detectar
from enlace_serial import EnlaceSerial
from main import ANCHO_PANEL, ALTO_VIDEO, AMARILLO, BLANCO, GRIS, ROJO, VERDE, abrir_fuente, nombre


class EstabilidadPorClase:
    """Una clase cuenta como presente si aparece en >= DET_VOTOS de los últimos N frames, con confianza media alta."""

    def __init__(self):
        self._hist = deque(maxlen=config.ESTABILIDAD_FRAMES)

    def agregar(self, detecciones) -> dict[str, float]:
        frame = {}
        for d in detecciones:
            frame[d.clase] = max(frame.get(d.clase, 0.0), d.confianza)
        self._hist.append(frame)
        estables = {}
        for clase in {c for h in self._hist for c in h}:
            confs = [h[clase] for h in self._hist if clase in h]
            if len(confs) >= config.DET_VOTOS and sum(confs) / len(confs) >= config.DET_CONF_ACTIVAR:
                estables[clase] = sum(confs) / len(confs)
        return estables

    def reiniciar(self):
        self._hist.clear()


def dibujar(frame, detecciones, estables, enlace, fps, envio_activo):
    h, w = frame.shape[:2]
    esc = ALTO_VIDEO / h
    video = cv2.resize(frame, (int(w * esc), ALTO_VIDEO))
    for d in detecciones:
        x0, y0, x1, y1 = (int(v * esc) for v in d.caja)
        color = VERDE if d.clase in estables else AMARILLO
        cv2.rectangle(video, (x0, y0), (x1, y1), color, 2)
        etiqueta = f"{nombre(d.clase)} {d.confianza * 100:.0f}%"
        cv2.rectangle(video, (x0, max(y0 - 20, 0)), (x0 + 9 * len(etiqueta), max(y0, 20)), color, -1)
        cv2.putText(video, etiqueta, (x0 + 3, max(y0 - 5, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    panel = np.full((ALTO_VIDEO, ANCHO_PANEL, 3), 30, np.uint8)
    cv2.putText(panel, "ToolLighthouse", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, BLANCO, 2)
    cv2.putText(panel, f"{fps:4.1f} FPS", (ANCHO_PANEL - 90, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, GRIS, 1)
    if estables:
        cv2.putText(panel, "Detectado", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, VERDE, 2)
        y = 100
        for clase, conf in sorted(estables.items(), key=lambda kv: kv[1], reverse=True):
            ubic = config.MAPA_UBICACION.get(clase)
            donde = "" if ubic in (None, "x") else f"  ->  Ubicacion {ubic}"
            cv2.putText(panel, f"{nombre(clase)} {conf * 100:.0f}%{donde}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLANCO, 1)
            y += 26
    else:
        cv2.putText(panel, "Sin componente", (16, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMARILLO, 2)

    y = ALTO_VIDEO - 80
    cv2.putText(panel, f"Serial: {enlace.estado}", (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLANCO, 1)
    cv2.putText(panel, f"Ultimo enviado: {enlace.ultimo_enviado or '-'}", (16, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, BLANCO, 1)
    cv2.putText(panel, f"Envio: {'ACTIVO' if envio_activo else 'PAUSADO (espacio)'}", (16, y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, VERDE if envio_activo else ROJO, 1)
    return np.hstack([video, panel])


def main():
    ap = argparse.ArgumentParser(description="ToolLighthouse (detección): cámara -> modelo -> Arduino")
    ap.add_argument("--camara", type=int)
    ap.add_argument("--video")
    ap.add_argument("--puerto")
    ap.add_argument("--sin-serial", action="store_true")
    args = ap.parse_args()

    cap = abrir_fuente(args)
    if not cap.isOpened():
        raise SystemExit("No se pudo abrir la cámara/video. Corre scripts/listar_camaras.py para ver los índices.")

    enlace = EnlaceSerial(args.puerto, simulado=args.sin_serial)
    estabilidad = EstabilidadPorClase()
    envio_activo = True
    ultimo_t, fps = time.time(), 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if args.video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                time.sleep(0.2)
                continue

            detecciones = detectar(frame)
            estables = estabilidad.agregar(detecciones)
            if envio_activo:
                # con la lista de objetos estables manda el de mayor confianza; sin ninguno, "nada"
                mejor = max(estables, key=estables.get) if estables else "nada"
                enlace.enviar_clase(mejor)

            ahora = time.time()
            fps = 0.9 * fps + 0.1 / max(ahora - ultimo_t, 1e-6)
            ultimo_t = ahora
            cv2.imshow("ToolLighthouse (deteccion)", dibujar(frame, detecciones, estables, enlace, fps, envio_activo))
            tecla = cv2.waitKey(1) & 0xFF
            if tecla in (ord("q"), 27):
                break
            if tecla == ord(" "):
                envio_activo = not envio_activo
            if tecla == ord("r"):
                estabilidad.reiniciar()
    finally:
        cap.release()
        enlace.cerrar()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
