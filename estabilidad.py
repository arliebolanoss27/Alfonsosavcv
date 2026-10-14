"""
Estabilidad de la predicción: promedia las probabilidades de los últimos N frames
y solo da una clase "estable" si el promedio supera el umbral de confianza.
"""

from collections import deque

import config


class Estabilizador:
    def __init__(self, ventana: int = config.ESTABILIDAD_FRAMES, umbral: float = config.CONFIDENCE_THRESHOLD):
        self.umbral = umbral
        self._historial: deque[dict[str, float]] = deque(maxlen=ventana)

    def agregar(self, probs: dict[str, float]) -> tuple[dict[str, float], str | None]:
        """
        Devuelve (probabilidades promedio, clase estable o None).
        La clase estable solo existe con la ventana llena y confianza media >= umbral.
        """
        self._historial.append(probs)
        n = len(self._historial)
        promedio = {c: sum(h[c] for h in self._historial) / n for c in probs}
        mejor = max(promedio, key=promedio.get)
        llena = n == self._historial.maxlen
        if llena and promedio[mejor] >= self.umbral:
            return promedio, mejor
        return promedio, None

    def reiniciar(self):
        self._historial.clear()
