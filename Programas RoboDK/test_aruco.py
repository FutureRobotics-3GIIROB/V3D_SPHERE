#!/usr/bin/env python
"""
test_aruco.py - Script de pruebas para el tracker de ArUcos

Este script permite:
- Seleccionar cámara (local o IP)
- Recordar la última cámara usada
- Lanzar el tracker en diferentes modos
- Mostrar ejes, cargar STL, activar debug

Uso:
    uv run test_aruco.py                    # Modo interactivo
    uv run test_aruco.py --debug            # Modo debug
    uv run test_aruco.py --obj ejes         # Mostrar ejes 3D
    uv run test_aruco.py --obj ruta/al.stl  # Cargar modelo STL
    uv run test_aruco.py --camera 0         # Usar cámara local 0
    uv run test_aruco.py --camera http://ip:port/video
"""

import argparse
import sys
from pathlib import Path

# Suprimir warnings antes de importar OpenCV
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"

try:
    import cv2
    cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
except Exception:
    pass

# Importar nuestra librería
from aruco_lib import (
    ArucoTracker,
    run_viewer,
    list_available_cameras,
    load_camera_config,
    save_camera_config,
    normalize_camera_url,
    DEFAULT_CAMERA_URL
)


def print_header():
    """Imprime cabecera del programa."""
    print()
    print("=" * 50)
    print("  🎯 ArUco Tracker - Test Utility")
    print("=" * 50)
    print()


def select_camera_interactive() -> str:
    """Selección interactiva de cámara."""
    print("Buscando cámaras disponibles...")
    local_cameras = list_available_cameras(max_devices=6)
    last_url = load_camera_config()
    
    print()
    print("╔════════════════════════════════════════════════╗")
    print("║           FUENTES DE VIDEO DISPONIBLES         ║")
    print("╠════════════════════════════════════════════════╣")
    
    # Mostrar cámaras locales
    if local_cameras:
        for i, cam_idx in enumerate(local_cameras, start=1):
            print(f"║  {i}. Cámara local (índice {cam_idx})")
    else:
        print("║  (No se detectaron cámaras locales)")
    
    print("║")
    print("║  i. Ingresar IP de cámara manualmente")
    print(f"║  Enter. Usar última: {last_url[:40]}...")
    print("╚════════════════════════════════════════════════╝")
    print()
    
    while True:
        choice = input("Selecciona opción (o pega URL directamente): ").strip()
        
        # Enter = usar última
        if not choice:
            print(f"→ Usando: {last_url}")
            return last_url
        
        # URL directa
        if choice.startswith("http://") or choice.startswith("https://"):
            url = choice
            save_camera_config(url)
            print(f"→ Usando URL: {url}")
            return url
        
        # IP sin protocolo
        if "." in choice or ":" in choice:
            url = normalize_camera_url(choice)
            save_camera_config(url)
            print(f"→ Usando URL: {url}")
            return url
        
        # Opción 'i' para ingresar IP
        if choice.lower() == "i":
            ip_input = input(f"IP/URL (Enter = {last_url}): ").strip()
            if ip_input:
                url = normalize_camera_url(ip_input)
                save_camera_config(url)
                print(f"→ Usando URL: {url}")
                return url
            return last_url
        
        # Número de cámara local
        if choice.isdigit() and local_cameras:
            idx = int(choice)
            if 1 <= idx <= len(local_cameras):
                cam = local_cameras[idx - 1]
                print(f"→ Usando cámara local: {cam}")
                return cam
        
        print("❌ Opción inválida. Intenta de nuevo.")


def select_mode_interactive() -> dict:
    """Selección interactiva de modo de ejecución."""
    print()
    print("╔════════════════════════════════════════════════╗")
    print("║              MODO DE EJECUCIÓN                 ║")
    print("╠════════════════════════════════════════════════╣")
    print("║  1. Normal (detección básica)                  ║")
    print("║  2. Con ejes 3D                                ║")
    print("║  3. Debug (coordenadas + info extra)           ║")
    print("║  4. Debug + Ejes                               ║")
    print("║  5. Cargar modelo STL                          ║")
    print("╚════════════════════════════════════════════════╝")
    print()
    
    choice = input("Selecciona modo [1]: ").strip() or "1"
    
    config = {
        "show_axes": False,
        "debug_mode": False,
        "stl_path": None
    }
    
    if choice == "2":
        config["show_axes"] = True
    elif choice == "3":
        config["debug_mode"] = True
    elif choice == "4":
        config["show_axes"] = True
        config["debug_mode"] = True
    elif choice == "5":
        stl_path = input("Ruta al archivo STL: ").strip()
        if stl_path and Path(stl_path).exists():
            config["stl_path"] = stl_path
            config["show_axes"] = True
        else:
            print("❌ Archivo no encontrado, continuando sin STL")
    
    return config


def run_interactive():
    """Modo interactivo completo."""
    print_header()
    
    # Seleccionar cámara
    camera = select_camera_interactive()
    
    # Seleccionar modo
    config = select_mode_interactive()
    
    print()
    print("=" * 50)
    print("  Iniciando tracker...")
    print("  Controles: q=salir, r=reset bolos, d=debug, a=ejes, s=captura")
    print("=" * 50)
    print()
    
    # Ejecutar visor
    run_viewer(
        camera_source=camera,
        show_axes=config["show_axes"],
        stl_path=config["stl_path"],
        debug_mode=config["debug_mode"]
    )


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(
        description="ArUco Tracker - Herramienta de pruebas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  uv run test_aruco.py                      Modo interactivo
  uv run test_aruco.py --debug              Modo debug
  uv run test_aruco.py --obj ejes           Mostrar ejes 3D
  uv run test_aruco.py --obj mi_modelo.stl  Cargar modelo STL
  uv run test_aruco.py --camera 0           Cámara local índice 0
  uv run test_aruco.py --camera http://192.168.1.100:8080/video
  uv run test_aruco.py --list-cameras       Listar cámaras disponibles

Sistema de ArUcos:
  ID 0: Referencia del mundo (QR)
  ID 1: Bola
  ID 2+: Bolos (se tumban al colisionar con la bola)
        """
    )
    
    parser.add_argument(
        "--camera", "-c",
        type=str,
        default=None,
        help="Fuente de cámara: índice local (0, 1...) o URL (http://ip:port/video)"
    )
    
    parser.add_argument(
        "--obj", "-o",
        type=str,
        default=None,
        help="Objeto a mostrar: 'ejes' para ejes 3D, o ruta a archivo STL"
    )
    
    parser.add_argument(
        "--debug", "-d",
        action="store_true",
        help="Activar modo debug (muestra coordenadas y info extra)"
    )
    
    parser.add_argument(
        "--list-cameras", "-l",
        action="store_true",
        help="Listar cámaras disponibles y salir"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Modo interactivo (selección por menú)"
    )
    
    args = parser.parse_args()
    
    # Listar cámaras
    if args.list_cameras:
        print_header()
        print("Buscando cámaras...")
        cameras = list_available_cameras(max_devices=10)
        
        if cameras:
            print(f"\n✓ Cámaras locales encontradas: {cameras}")
            for cam in cameras:
                print(f"  - Índice {cam}")
        else:
            print("\n✗ No se encontraron cámaras locales")
        
        print(f"\nÚltima URL usada: {load_camera_config()}")
        return
    
    # Modo interactivo
    if args.interactive or (args.camera is None and args.obj is None and not args.debug):
        run_interactive()
        return
    
    # Modo con argumentos
    print_header()
    
    # Determinar cámara
    if args.camera is not None:
        if args.camera.isdigit():
            camera = int(args.camera)
        else:
            camera = normalize_camera_url(args.camera)
            save_camera_config(camera)
    else:
        camera = load_camera_config()
    
    # Determinar opciones
    show_axes = False
    stl_path = None
    
    if args.obj:
        if args.obj.lower() in ["ejes", "axes", "axis"]:
            show_axes = True
        elif Path(args.obj).exists():
            stl_path = args.obj
            show_axes = True
        else:
            print(f"⚠ Archivo no encontrado: {args.obj}")
            print("  Continuando con ejes activados...")
            show_axes = True
    
    print(f"Cámara: {camera}")
    print(f"Debug: {'Sí' if args.debug else 'No'}")
    print(f"Ejes: {'Sí' if show_axes else 'No'}")
    if stl_path:
        print(f"STL: {stl_path}")
    print()
    
    # Ejecutar
    run_viewer(
        camera_source=camera,
        show_axes=show_axes,
        stl_path=stl_path,
        debug_mode=args.debug
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[Interrumpido por usuario]")
        sys.exit(0)
