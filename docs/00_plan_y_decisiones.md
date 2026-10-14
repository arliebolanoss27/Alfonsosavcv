# Plan del proyecto y decisiones

Reto ToolLighthouse, Semana KEY: un sistema que reconoce un componente frente a una cámara e ilumina el LED de su ubicación en un estante.

```
Objeto -> Cámara (iPhone) -> Visión artificial -> Python -> Arduino Nano -> LED de la ubicación
```

Este documento reúne lo decidido, lo pendiente y el plan de trabajo del **modelo de visión**. La electrónica y el estante los lleva otro integrante.

## 1. Entregables de documentación

| Entregable | Archivo |
|---|---|
| Selección del modelo y su justificación | [`01_seleccion_modelo.md`](01_seleccion_modelo.md) |
| Descripción del dataset | [`02_descripcion_dataset.md`](02_descripcion_dataset.md) |

## 2. Estado (13 de octubre, día 2)

| Parte | Estado |
|---|---|
| Selección del modelo | Hecha (YOLO26-cls), falta confirmarla con validación en datos propios |
| Descripción del dataset | Borrador con los datos públicos revisados; faltan fotos y videos propios y la curación de Kaggle |
| Código de preparación de datos | No escrito |
| Entrenamiento | No hecho |
| Integración cámara -> modelo -> serial | No escrita |

No hay código en el repositorio todavía.

## 3. Decisiones tomadas

| Tema | Decisión |
|---|---|
| Tarea | Clasificación de imagen, 10 clases (9 componentes + `nada`) |
| Modelo | YOLO26s-cls con transfer learning; YOLO26m-cls como comparación |
| Entrenamiento | En la laptop del equipo (RTX 4070); Colab como respaldo |
| Nombres de clase | `resistencia`, `diodo`, `led`, `potenciometro`, `jumper`, `servo`, `protoboard`, `arduino_nano`, `capacitor_electrolitico`, `nada` |
| Potenciómetro | Solo el de perilla (taper); el trimmer queda fuera |
| Variaciones de imagen | Augmentation al vuelo durante el entrenamiento, sin cambiar el tono de color. No se guardan variaciones en disco |
| Split | 80 % / 20 % por grupo de origen (video o sesión de fotos), nunca por imagen suelta |
| Umbral de confianza | 0.80 |
| Estabilidad | Promedio de probabilidades de los últimos ~8 frames, en vez de exigir N frames seguidos |
| Interfaz del modelo | `predecir(frame) -> (clase, confianza)` |
| Cámara | iPhone 16 como webcam por **cable USB** (DroidCam o Iriun), enfoque y exposición bloqueados |
| Montaje | Cámara fija sobre un fondo gris mate con luz constante |
| Clase `nada` | Apaga todos los LEDs (ver sección 5) |
| Commits | Un commit por acción, con el porqué; los hace y sube el equipo a mano |

## 4. Datos

Resumen; el detalle está en [`02_descripcion_dataset.md`](02_descripcion_dataset.md).

- **ElectroCom-61** (Mendeley, CC BY 4.0): fotos reales de laboratorio con cajas. Se recorta cada caja. Es la fuente principal de calidad.
- **Electronic Component Recognizer** (Kaggle, MIT): muy ruidoso (diagramas, logos, personas, suéteres en `jumper`). **No se usa sin filtrar.**
- **Fotos y videos propios:** 5 videos de ~15 s por clase con el iPhone 16, más 5 de `nada`. Son la fuente más importante, porque se parecen a la demo.
- Los datasets públicos crudos se suben al repositorio en `datasets/` para el traspaso. Qué falta: [`03_pendientes_modelo_y_camara.md`](03_pendientes_modelo_y_camara.md).

## 5. Comunicación con la electrónica

El estante tiene un Arduino Nano con 9 LEDs en los pines D2 a D10.

- Firmware actual, a **115200 baudios**: `'1'` a `'9'` enciende solo esa ubicación; `'0'` vuelve a la secuencia de prueba; cualquier otro carácter se ignora. D0 y D1 están reservados para el serial.
- **Cambio propuesto al firmware, aún sin confirmar con quien hace la electrónica:** que `'x'` apague los 9 LEDs. Sin ese comando, al retirar el objeto el último LED se quedaría encendido.
- Python envía el carácter solo cuando la clase estable **cambia**, para no saturar el serial.

## 6. Plan de trabajo

**Día 2 (hoy): dejar el modelo entrenado**
1. Estructura del repo y `config.py` (clases, umbral, mapa clase -> ubicación, puerto serial).
2. Script que recorta ElectroCom-61 hacia las clases propias.
3. Script que extrae frames de los videos, con el split por video.
4. Filtro de Kaggle: duplicados exactos y revisión de las imágenes de baja confianza.
5. Revisión visual de una muestra por clase.
6. Entrenamiento (YOLO26s-cls y m) y evaluación: exactitud, matriz de confusión y prueba solo con fotos propias.

**Día 3: integrar**
7. `clasificador.py` con `predecir(frame)`.
8. `main.py`: cámara -> recorte de la zona marcada -> modelo -> filtro de estabilidad -> serial.
9. Pruebas de punta a punta. Si una clase falla, más videos de esa clase y reentrenar.

**Día 4: demostrar**
10. Ajuste fino de umbral y estabilidad, ensayo de la demo y documentación final.

## 7. Pendientes

| Pendiente | Quién |
|---|---|
| Mapa clase -> ubicación (1 a 9) del estante | Equipo, junto con electrónica |
| Confirmar el comando `'x'` en el firmware | Electrónica |
| Fotos de referencia propias | Equipo |
| Videos con el iPhone 16 sobre el montaje | Equipo, cuando esté el montaje |
| Curación de las imágenes de Kaggle | Modelo |
| Confirmar que el entorno de Python de la laptop tiene GPU funcionando | Modelo |
| Probar que el iPhone se detecta como cámara por USB en Windows | Equipo |

## 8. Riesgos

- **Domain gap:** los datos públicos vienen de otras cámaras, mesas y luces. Se mide también la exactitud solo con fotos propias.
- **Poca diversidad** en resistencia, diodo y LED; dependen de los videos propios.
- **Pares confundibles:** resistencia y diodo, jumper y otros cables.
- **Objetos delgados:** hay que sostenerlos cerca de la cámara.
- **Cámara en demo:** llevar una webcam USB de respaldo por si el teléfono falla.
- **Si el Nano da problemas** de reconocimiento, se cambia por Arduino Uno.
