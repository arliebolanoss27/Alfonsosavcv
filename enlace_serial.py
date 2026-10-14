"""
Envío de la ubicación al Arduino Nano por serial.
Si no hay Nano conectado (o se pide --sin-serial) funciona en modo simulado:
no escribe en ningún puerto, solo registra lo que enviaría.
"""

import time

import config


class EnlaceSerial:
    def __init__(self, puerto: str | None = None, simulado: bool = False):
        self.puerto = puerto or config.SERIAL_PORT
        self.simulado = simulado
        self.ultimo_enviado: str | None = None
        self.estado = "simulado"
        self._ser = None
        if not simulado:
            self._abrir()

    def _abrir(self):
        try:
            import serial
            self._ser = serial.Serial(self.puerto, config.SERIAL_BAUDRATE, timeout=config.SERIAL_TIMEOUT)
            time.sleep(config.SERIAL_ESPERA_RESET)  # el Nano se reinicia al abrir el puerto
            self.estado = f"conectado {self.puerto}"
        except Exception as e:  # puerto inexistente, ocupado, pyserial ausente...
            print(f"[serial] No se pudo abrir {self.puerto} ({e}). Modo simulado.")
            self.simulado = True
            self.estado = f"simulado ({self.puerto} no disponible)"

    def enviar_clase(self, clase: str) -> str | None:
        """
        Envía la ubicación de la clase solo si cambió respecto a la última enviada.
        Devuelve el carácter enviado, o None si no se envió nada.
        """
        caracter = config.MAPA_UBICACION.get(clase)
        if caracter is None:
            return None
        if caracter == "x" and not config.SOPORTA_APAGADO:
            return None
        if caracter == self.ultimo_enviado:
            return None
        if self._ser is not None:
            self._ser.write(caracter.encode("ascii"))
        print(f"[serial] {'(simulado) ' if self.simulado else ''}{clase} -> '{caracter}'")
        self.ultimo_enviado = caracter
        return caracter

    def apuntar(self, clase: str) -> bool:
        """Apunta el láser a la caja de la clase. Mientras el protocolo no esté acordado solo lo imprime."""
        n = config.MAPA_UBICACION.get(clase)
        if n in (None, "x"):
            return False
        self._enviar_laser(config.LASER_COMANDO.format(n=n), f"láser -> caja {n} ({clase})")
        return True

    def apagar_laser(self):
        self._enviar_laser(config.LASER_APAGAR, "láser apagado")

    def _enviar_laser(self, comando: str, descripcion: str):
        real = config.LASER_ACTIVO and self._ser is not None
        if real:
            self._ser.write(comando.encode("ascii"))
        print(f"[serial] {'' if real else '(simulado) '}{descripcion}  {comando!r}")

    def cerrar(self):
        if self._ser is not None:
            self._ser.close()
