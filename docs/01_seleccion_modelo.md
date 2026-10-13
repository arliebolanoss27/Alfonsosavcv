# Selección del modelo

**Decisión:** clasificación de imagen con **YOLO26-cls** (Ultralytics), usando transfer learning desde pesos preentrenados en ImageNet. Se entrena la variante **YOLO26s-cls** como modelo principal y **YOLO26m-cls** como comparación; se queda la que mejor resulte en validación con fotos propias.

## 1. Tarea

El reto pide que el sistema reconozca qué objeto hay frente a la cámara y encienda la ubicación correspondiente. Hay un solo objeto por escena, así que la tarea es **clasificación** (una etiqueta por imagen entre 10 clases: 9 componentes más "nada"). La detección (cajas por objeto) es opcional en el reto y no aporta aquí: no necesitamos saber *dónde* está el objeto, solo *cuál* es.

## 2. Opciones consideradas

| Opción | Qué es | Por qué sí / por qué no |
|---|---|---|
| MobileNetV2 (Keras) | Red ligera clásica, mencionada en el material del reto | Funciona, pero habría que escribir a mano el entrenamiento, la augmentation, las métricas y la matriz de confusión |
| YOLOv8n-cls | Opción alternativa del material del reto | Es una generación anterior; Ultralytics ya tiene una más nueva con el mismo flujo |
| **YOLO26-cls** | Generación actual de clasificación de Ultralytics | **Elegida** (justificación abajo) |
| Modelo previo de 130+ clases de un compañero | Evaluado antes | Descartado: no aporta ventaja sobre uno nuevo ajustado a nuestras 10 clases |

## 3. Variantes de YOLO26-cls

Datos de la documentación oficial de Ultralytics (entrada de 224 px, preentrenado en ImageNet):

| Variante | Top-1 en ImageNet | Parámetros | CPU (ONNX) | GPU T4 (TensorRT) |
|---|---|---|---|---|
| YOLO26n-cls | 71.4 % | 2.8 M | 5.0 ms | 1.1 ms |
| **YOLO26s-cls** | **76.0 %** | **6.7 M** | **7.9 ms** | **1.3 ms** |
| YOLO26m-cls | 78.1 % | 11.6 M | 17.2 ms | 2.0 ms |

Fuente: [Ultralytics Docs, Image Classification](https://docs.ultralytics.com/tasks/classify).

La variante **s** es el punto de partida: gana casi 5 puntos de exactitud sobre la **n** y sigue tardando menos de 10 ms por imagen en CPU. La **m** se entrena también porque entrenar cuesta minutos y el costo en inferencia sigue siendo bajo; elegimos entre las dos con **nuestros** datos, no con ImageNet.

## 4. Justificación

1. **Encaja con la tarea.** Es un clasificador de una etiqueta por imagen, que es lo que pide la presentación del reto.
2. **Transfer learning.** Partimos de pesos preentrenados y solo ajustamos a nuestras clases. Con unos cientos de imágenes por clase esto rinde mucho más que entrenar desde cero.
3. **Velocidad suficiente para "ver, decidir y actuar".** Con unos 8 ms por imagen en CPU, el modelo no es el cuello de botella; la cámara y el serial limitan antes. La demo corre en una laptop con GPU RTX 4070, con holgura de sobra.
4. **Un solo flujo de trabajo.** Ultralytics resuelve en la misma herramienta el entrenamiento, la augmentation (voltear, rotar, variar brillo, como pide el material del reto), las métricas en validación, la matriz de confusión y la inferencia. Menos código propio es menos errores y un montaje más "ordenado y bien documentado" (criterio de calidad técnica).
5. **Interfaz estable.** El modelo se oculta detrás de una función `predecir(frame) -> (clase, confianza)`. Si hubiera que cambiar de arquitectura, el script de la cámara no cambia.
6. **Riesgo controlado.** Si YOLO26 diera problemas, el plan B es YOLO11/YOLOv8-cls o MobileNetV2 con la misma interfaz.

## 5. Decisiones de entrenamiento

- **Entrada:** 224×224 px.
- **Augmentation al vuelo** (en cada época), no variaciones guardadas en disco: así no se multiplican archivos y se evita la fuga de datos entre train y validación.
- **Sin cambios de tono de color.** Los colores de las bandas de las resistencias y de los LEDs son parte de lo que define la clase.
- **Split por grupo de origen** (por video o por sesión de fotos) y nunca por imagen suelta, para que la validación no se infle.
- **Clase "nada"** incluida para que el modelo no elija siempre uno de los 9 objetos cuando no hay ninguno.

## 6. Cómo se confirma la elección

La arquitectura se da por buena si cumple, en validación con **fotos y videos de nuestra cámara** (no solo con datasets públicos):

- exactitud global y por clase aceptable, revisada en la **matriz de confusión** (pares de riesgo: resistencia↔diodo, jumper↔otros cables);
- confianza por encima del umbral de **0.80** en los objetos correctos y por debajo en "nada";
- latencia compatible con la demo (decenas de milisegundos por imagen).

El resultado de esa comparación se agrega a este documento cuando termine el entrenamiento.

## 7. Limitaciones

- Los números de ImageNet de la tabla son de referencia, no predicen nuestra exactitud.
- Ultralytics se distribuye bajo licencia AGPL-3.0: sin problema para un prototipo académico, pero a tener en cuenta si el proyecto se comercializara.
