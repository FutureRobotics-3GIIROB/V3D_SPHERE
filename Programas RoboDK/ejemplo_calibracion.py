#!/usr/bin/env python
"""
ejemplo_calibracion.py - Ejemplos de uso del sistema de calibración.
"""

from calibracion import HomographyCalibrator
import numpy as np
import json


def ejemplo_1_cargar():
    """Ejemplo 1: Cargar calibración guardada."""
    print("\n" + "=" * 60)
    print("EJEMPLO 1: Cargar Calibración")
    print("=" * 60)
    
    calib_data = HomographyCalibrator.load_calibration()
    
    if calib_data is None:
        print("\n❌ No se encontró calibración")
        print("   Ejecuta: python calibracion.py --capture\n")
        return
    
    print("\n✓ Calibración cargada exitosamente")
    print(f"\n  Ajedrez: {calib_data['chessboard_size']}")
    print(f"  Cuadro: {calib_data['square_size_mm']} mm")
    print(f"  Archivo: {HomographyCalibrator.CALIBRATION_FILE}\n")


def ejemplo_2_transformar():
    """Ejemplo 2: Transformar puntos de píxeles a patrón."""
    print("\n" + "=" * 60)
    print("EJEMPLO 2: Transformar Puntos")
    print("=" * 60)
    
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("\n❌ No se encontró calibración\n")
        return
    
    homography = calib_data['homography_matrix']
    
    # Puntos de prueba
    test_points = [
        (100, 100),
        (320, 240),
        (640, 480),
    ]
    
    print("\nTransformaciones:")
    for px in test_points:
        x_mm, y_mm = HomographyCalibrator.transform_point(px, homography)
        print(f"  Píxel {px} → Patrón ({x_mm:.1f}, {y_mm:.1f}) mm")
    
    print()


def ejemplo_3_json():
    """Ejemplo 3: Crear estructura JSON para salida."""
    print("\n" + "=" * 60)
    print("EJEMPLO 3: Estructura JSON de Salida")
    print("=" * 60)
    
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("\n❌ No se encontró calibración\n")
        return
    
    homography = calib_data['homography_matrix']
    
    # Simular detección de pelota
    ball_pixel = (256, 200)
    x_mm, y_mm = HomographyCalibrator.transform_point(ball_pixel, homography)
    
    # Crear estructura JSON
    ball_position = {
        "pixel": list(ball_pixel),
        "xyz_mm": [x_mm, y_mm, 0]
    }
    
    print("\nPelota detectada en píxeles:", ball_pixel)
    print("\nEstructura JSON:")
    print(json.dumps(ball_position, indent=2))
    print()


def ejemplo_4_estadisticas():
    """Ejemplo 4: Estadísticas de calibración."""
    print("\n" + "=" * 60)
    print("EJEMPLO 4: Estadísticas de Calibración")
    print("=" * 60)
    
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("\n❌ No se encontró calibración\n")
        return
    
    cam_matrix = np.array(calib_data['camera_matrix'])
    h_matrix = np.array(calib_data['homography_matrix'])
    
    print("\nMatriz de Cámara:")
    print(f"  Focal X: {cam_matrix[0, 0]:.2f} px")
    print(f"  Focal Y: {cam_matrix[1, 1]:.2f} px")
    print(f"  Centro X: {cam_matrix[0, 2]:.2f} px")
    print(f"  Centro Y: {cam_matrix[1, 2]:.2f} px")
    
    print("\nMatriz de Homografía:")
    print(f"  Determinante: {np.linalg.det(h_matrix):.4f}")
    print(f"  Número de condición: {np.linalg.cond(h_matrix):.4f}")
    print()


def ejemplo_5_lote():
    """Ejemplo 5: Procesar lote de detecciones."""
    print("\n" + "=" * 60)
    print("EJEMPLO 5: Procesamiento en Lote")
    print("=" * 60)
    
    calib_data = HomographyCalibrator.load_calibration()
    if calib_data is None:
        print("\n❌ No se encontró calibración\n")
        return
    
    homography = calib_data['homography_matrix']
    
    # Simular detecciones a lo largo del tiempo
    detecciones = [
        (100, 100),
        (110, 105),
        (120, 110),
        (125, 115),
        (130, 120),
    ]
    
    print(f"\nProcesando {len(detecciones)} detecciones:\n")
    
    posiciones = []
    for i, px in enumerate(detecciones, 1):
        x_mm, y_mm = HomographyCalibrator.transform_point(px, homography)
        posiciones.append((x_mm, y_mm))
        print(f"  Frame {i}: píxel {px} → ({x_mm:.1f}, {y_mm:.1f}) mm")
    
    # Calcular promedio
    x_promedio = np.mean([p[0] for p in posiciones])
    y_promedio = np.mean([p[1] for p in posiciones])
    
    print(f"\nPosición promedio: ({x_promedio:.1f}, {y_promedio:.1f}) mm")
    print()


def main():
    """Ejecutar todos los ejemplos."""
    print("\n" + "=" * 60)
    print("EJEMPLOS DE USO - CALIBRACIÓN POR HOMOGRAFÍA")
    print("=" * 60)
    
    ejemplo_1_cargar()
    ejemplo_2_transformar()
    ejemplo_3_json()
    ejemplo_4_estadisticas()
    ejemplo_5_lote()
    
    print("=" * 60)
    print("Ejemplos completados")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
