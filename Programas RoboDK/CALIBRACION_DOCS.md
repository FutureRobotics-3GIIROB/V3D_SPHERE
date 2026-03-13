# Sistema de Calibración por Homografía - Documentación Completa

## 📋 Resumen de Cambios

Se ha implementado un nuevo sistema de **calibración por homografía** usando un patrón de ajedrez 6×9, reemplazando completamente el anterior sistema de calibración por QR.

### Cambios Principales:

1. **Nuevo archivo: `calibracion.py`** ✓
   - Captura 3 imágenes de patrón de ajedrez
   - Calcula matriz de homografía
   - Guarda en `calibration_data.json`

2. **Modificado: `main.py`** ✓
   - Eliminadas todas referencias a `qr_depth.py`
   - Importa `HomographyCalibrator`
   - Usa transformación por homografía

3. **Modificado: `main_robot.py`** ✓
   - Eliminadas todas referencias a QR
   - Integración con calibración por homografía
   - Mismo funcionamiento con nueva transformación

## 🚀 Uso Rápido

### Paso 1: Calibrar la Cámara

```bash
python calibracion.py --capture
```

**Instrucciones:**
1. Coloca el patrón de ajedrez frente a la cámara
2. Presiona **ESPACIO** cuando el patrón sea detectado (se muestra en verde)
3. Repite para 3 imágenes desde ángulos diferentes
4. Se generará `calibration_data.json`

### Paso 2: Ejecutar la Aplicación Principal

```bash
python main.py
```

O para control de robot:

```bash
python main_robot.py
```

### Paso 3: Verificar Calibración

```bash
python calibracion.py --load
```

Muestra los parámetros de calibración guardados.

## 📊 Especificaciones del Patrón

- **Dimensión interna**: 6 × 9 esquinas
- **Tamaño de cuadro**: 30 mm
- **Material recomendado**: Papel blanco/negro de alta definición

## 📄 Archivo de Calibración (`calibration_data.json`)

```json
{
  "camera_matrix": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "homography_matrix": [[h11, h12, h13], [h21, h22, h23], [h31, h32, h33]],
  "chessboard_size": [6, 9],
  "square_size_mm": 30.0,
  "notes": "Calibración por patrón de ajedrez"
}
```

## 🔄 Flujo de Transformación

```
Píxeles en imagen (px, py)
         ↓
    Matriz de Homografía
         ↓
Coordenadas del patrón (x_mm, y_mm)
         ↓
Salida JSON: {"pixel": [px, py], "xyz_mm": [x, y, 0]}
```

## 💻 API de `HomographyCalibrator`

### Métodos Principales

```python
from calibracion import HomographyCalibrator

# Cargar calibración guardada
calib_data = HomographyCalibrator.load_calibration()
if calib_data:
    homography = calib_data['homography_matrix']
    
    # Transformar un punto
    point_px = (100, 200)
    point_mm = HomographyCalibrator.transform_point(point_px, homography)
    print(f"Píxel {point_px} → Patrón {point_mm}")
```

### Atributos de Clase

```python
HomographyCalibrator.CHESSBOARD_SIZE = (6, 9)      # Dimensión del patrón
HomographyCalibrator.SQUARE_SIZE_MM = 30.0         # Tamaño del cuadro
HomographyCalibrator.CALIBRATION_FILE = "calibration_data.json"
```

## 🔧 Opciones de Línea de Comando

### `calibracion.py`

```bash
# Capturar nuevas imágenes
python calibracion.py --capture

# Cargar y mostrar parámetros
python calibracion.py --load

# Usar cámara específica
python calibracion.py --capture --camera 1

# Alias cortos
python calibracion.py -c          # capturar
python calibracion.py -l          # cargar
```

### `main.py`

```bash
# Ejecutar con modo por defecto
python main.py
```

### `main_robot.py`

```bash
# Integración con RoboDK
python main_robot.py
```

## ⚠️ Troubleshooting

### Error: "No se encontró archivo de calibración"

**Solución:**
```bash
python calibracion.py --capture
```

Asegúrate de completar el proceso de captura de 3 imágenes.

### No se detecta el patrón de ajedrez

**Checklist:**
- ✓ Iluminación adecuada y uniforme
- ✓ Patrón completamente visible en la imagen
- ✓ Patrón tiene buen contraste (blanco y negro nítidos)
- ✓ Distancia cámara-patrón: 30-60 cm
- ✓ Intenta desde diferentes ángulos

### Coordenadas transformadas incorrectas

**Soluciones:**
1. Vuelve a ejecutar calibración con mejor calidad
2. Usa 3 imágenes desde ángulos significativamente diferentes
3. Asegúrate de que todo el patrón sea visible
4. Verifica que el patrón mida exactamente 30mm por cuadro

## 🗑️ Referencias Eliminadas

Completamente removidas del código:
- ❌ `qr_depth.py` - módulo de profundidad por QR
- ❌ `QRDepth` - clase de detección de QR
- ❌ `focal_length()` - función de cálculo focal
- ❌ `pixel_to_xyz()` - función de conversión píxel a 3D
- ❌ `QR_WORLD_REFERENCE_ID` - constante de QR
- ❌ Todas las referencias a `frame_data.qr_detections`

## 📤 Formato de Salida

La posición de la pelota se guarda en `positions.json`:

```json
{
  "pixel": [320, 240],
  "xyz_mm": [45.2, 67.8, 0]
}
```

Donde:
- `pixel`: Coordenadas en píxeles [x, y]
- `xyz_mm`: Coordenadas en mm del patrón de ajedrez
  - `x`, `y`: Transformadas por homografía (en mm)
  - `z`: 0 (la pelota está en el plano del patrón)

## 🔬 Ejemplo de Uso Avanzado

```python
from calibracion import HomographyCalibrator
import cv2

# Cargar calibración
calib = HomographyCalibrator.load_calibration()
if not calib:
    print("Ejecuta: python calibracion.py --capture")
    exit()

homography = calib['homography_matrix']

# Procesar video
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Tu lógica de detección...
    detected_pixel = (100, 150)  # Ejemplo
    
    # Transformar a coordenadas del patrón
    x_mm, y_mm = HomographyCalibrator.transform_point(
        detected_pixel, homography
    )
    
    print(f"Píxel {detected_pixel} → Patrón ({x_mm:.1f}, {y_mm:.1f}) mm")
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
```

## 📝 Notas Técnicas

### Matriz de Homografía

La matriz de homografía H transforma coordenadas de imagen a coordenadas del patrón:

```
[x']   [h11 h12 h13] [px]
[y'] = [h21 h22 h23] [py]
[w ]   [h31 h32 h33] [1 ]

x = x' / w
y = y' / w
```

### Calibración en Múltiples Ángulos

Se capturan 3 imágenes desde ángulos diferentes para:
- Mejorar robustez
- Reducir errores de perspectiva
- Obtener matriz promediada más precisa

## 🎯 Casos de Uso

✅ **Tracking de objetos con coordenadas 2D calibradas**
✅ **Control de robot basado en visión**
✅ **Medición de distancias en plano**
✅ **Detección de posición de pelota/objetos**

## 🚫 Limitaciones Conocidas

⚠️ Z siempre es 0 (transformación 2D del plano del patrón)
⚠️ Requiere patrón visible para calibración inicial
⚠️ La precisión depende de calidad de captura

## ✅ Checklist de Implementación

- [x] Crear `calibracion.py` funcional
- [x] Eliminar referencias a QR en `main.py`
- [x] Eliminar referencias a QR en `main_robot.py`
- [x] Integrar `HomographyCalibrator` en mains
- [x] Documentación completa
- [x] Sin errores de compilación

## 📞 Soporte

Para problemas o preguntas:
1. Verifica el archivo `calibration_data.json` existe
2. Ejecuta `python calibracion.py --load` para ver parámetros
3. Prueba con `python calibracion.py --capture` nuevamente
