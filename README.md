# Alfonsosavcv

Proyecto hecho para la semana KEY, el cual va dirigido a la automatización del reconocimiento de los instrumentos en el laboratorio.

Una persona muestra un componente a una cámara; un modelo de visión lo clasifica y el sistema enciende, por serial, el LED de su ubicación en un estante.

```
Objeto -> Cámara (iPhone) -> Visión artificial -> Python -> Arduino Nano -> LED
```

## Documentación

| Documento | Contenido |
|---|---|
| [`docs/00_plan_y_decisiones.md`](docs/00_plan_y_decisiones.md) | Estado, decisiones, plan de trabajo, pendientes y riesgos |
| [`docs/01_seleccion_modelo.md`](docs/01_seleccion_modelo.md) | Selección del modelo (YOLO26-cls) y su justificación |
| [`docs/02_descripcion_dataset.md`](docs/02_descripcion_dataset.md) | Descripción del dataset: clases, fuentes, cantidades y calidad |
| [`docs/03_pendientes_modelo_y_camara.md`](docs/03_pendientes_modelo_y_camara.md) | **Traspaso:** qué falta del modelo y cómo conectar cámara, modelo y Arduino |
| [`docs/contexto_original_del_reto.md`](docs/contexto_original_del_reto.md) | Contexto original del reto y decisiones iniciales |

## Clases

`resistencia`, `diodo`, `led`, `potenciometro`, `jumper`, `servo`, `protoboard`, `arduino_nano`, `capacitor_electrolitico` y `nada`.

## Estado

En desarrollo. Hasta ahora solo existe la documentación; el código de preparación de datos, entrenamiento e integración está por escribir.

## Datos

Los datasets públicos están en `datasets/` tal como se descargaron, sin procesar: `datasets/electrocom61/` (ElectroCom-61, CC BY 4.0) y `datasets/kaggle_recognizer/` (Electronic Component Recognizer, MIT, ruidoso). Ver `docs/02_descripcion_dataset.md` y `docs/03_pendientes_modelo_y_camara.md`.
