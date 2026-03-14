#!/usr/bin/env python
"""
calibracion.py - Calibracion por homografia con patron de ajedrez.

Uso:
  python calibracion.py --capture
  python calibracion.py --load
  python calibracion.py --capture --camera 1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


class HomographyCalibrator:
	"""Calibrador de camara y homografia para un tablero de ajedrez."""

	CHESSBOARD_SIZE: Tuple[int, int] = (6, 9)
	SQUARE_SIZE_MM: float = 30.0
	REQUIRED_CAPTURES: int = 3
	CALIBRATION_FILE: str = "calibration_data.json"

	@classmethod
	def _module_dir(cls) -> Path:
		return Path(__file__).resolve().parent

	@classmethod
	def _save_path(cls) -> Path:
		return cls._module_dir() / cls.CALIBRATION_FILE

	@classmethod
	def _read_candidate_paths(cls) -> List[Path]:
		# Compatibilidad: buscar tanto en cwd como junto al script.
		cwd_path = Path.cwd() / cls.CALIBRATION_FILE
		module_path = cls._module_dir() / cls.CALIBRATION_FILE
		if cwd_path.resolve() == module_path.resolve():
			return [cwd_path]
		return [cwd_path, module_path]

	@classmethod
	def _pattern_object_points(cls) -> np.ndarray:
		"""Genera puntos del tablero en mm en el plano Z=0."""
		cols, rows = cls.CHESSBOARD_SIZE
		objp = np.zeros((rows * cols, 3), np.float32)
		grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
		objp[:, :2] = grid * cls.SQUARE_SIZE_MM
		return objp

	@classmethod
	def save_calibration(
		cls,
		camera_matrix: np.ndarray,
		homography_matrix: np.ndarray,
		dist_coeffs: Optional[np.ndarray] = None,
		reprojection_error: Optional[float] = None,
	) -> Path:
		"""Guarda calibracion a JSON y devuelve la ruta."""
		payload: Dict[str, object] = {
			"camera_matrix": np.asarray(camera_matrix, dtype=float).tolist(),
			"homography_matrix": np.asarray(homography_matrix, dtype=float).tolist(),
			"chessboard_size": list(cls.CHESSBOARD_SIZE),
			"square_size_mm": float(cls.SQUARE_SIZE_MM),
			"notes": "Calibracion por patron de ajedrez",
		}
		if dist_coeffs is not None:
			payload["distortion_coefficients"] = (
				np.asarray(dist_coeffs, dtype=float).reshape(-1).tolist()
			)
		if reprojection_error is not None:
			payload["reprojection_error"] = float(reprojection_error)

		out_path = cls._save_path()
		out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
		return out_path

	@classmethod
	def load_calibration(cls) -> Optional[Dict[str, object]]:
		"""Carga calibracion guardada, si existe y es valida."""
		for path in cls._read_candidate_paths():
			if not path.exists():
				continue

			try:
				data = json.loads(path.read_text(encoding="utf-8"))
			except (json.JSONDecodeError, OSError):
				continue

			if "homography_matrix" not in data:
				continue
			if "camera_matrix" not in data:
				continue

			# Devolver matrices como np.ndarray para facilitar operaciones.
			data["camera_matrix"] = np.asarray(data["camera_matrix"], dtype=np.float64)
			data["homography_matrix"] = np.asarray(
				data["homography_matrix"], dtype=np.float64
			)
			if "distortion_coefficients" in data:
				data["distortion_coefficients"] = np.asarray(
					data["distortion_coefficients"], dtype=np.float64
				)
			return data

		return None

	@staticmethod
	def transform_point(
		point_px: Sequence[float], homography_matrix: Sequence[Sequence[float]]
	) -> Tuple[float, float]:
		"""Transforma un punto (x,y) de pixel a coordenadas del patron (mm)."""
		h = np.asarray(homography_matrix, dtype=np.float64)
		p = np.array([[[float(point_px[0]), float(point_px[1])]]], dtype=np.float64)
		transformed = cv2.perspectiveTransform(p, h)[0, 0]
		return float(transformed[0]), float(transformed[1])

	@classmethod
	def capture_calibration(
		cls, camera_index: int = 0
	) -> Optional[Dict[str, np.ndarray]]:
		"""Captura imagenes del tablero y calcula la calibracion."""
		cap = cv2.VideoCapture(camera_index)
		if not cap.isOpened():
			print(f"[ERROR] No se pudo abrir la camara {camera_index}")
			return None

		criteria = (
			cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
			30,
			0.001,
		)
		objp = cls._pattern_object_points()

		object_points: List[np.ndarray] = []
		image_points: List[np.ndarray] = []

		print("[INFO] Captura de calibracion iniciada")
		print(f"[INFO] Tablero esperado: {cls.CHESSBOARD_SIZE[0]}x{cls.CHESSBOARD_SIZE[1]}")
		print(f"[INFO] Capturas requeridas: {cls.REQUIRED_CAPTURES}")
		print("[INFO] Controles: ESPACIO=guardar captura, ESC=cancelar")

		last_gray_shape: Optional[Tuple[int, int]] = None

		try:
			while len(image_points) < cls.REQUIRED_CAPTURES:
				ok, frame = cap.read()
				if not ok:
					print("[WARN] No se pudo leer frame de la camara")
					continue

				gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
				last_gray_shape = gray.shape[::-1]

				found, corners = cv2.findChessboardCorners(gray, cls.CHESSBOARD_SIZE)

				draw = frame.copy()
				if found:
					refined = cv2.cornerSubPix(
						gray,
						corners,
						(11, 11),
						(-1, -1),
						criteria,
					)
					cv2.drawChessboardCorners(draw, cls.CHESSBOARD_SIZE, refined, found)

					status = (
						f"Detectado ({len(image_points)}/{cls.REQUIRED_CAPTURES}) - "
						"Presiona ESPACIO"
					)
					color = (0, 200, 0)
				else:
					refined = None
					status = (
						f"Busca tablero ({len(image_points)}/{cls.REQUIRED_CAPTURES})"
					)
					color = (0, 0, 255)

				cv2.putText(
					draw,
					status,
					(10, 30),
					cv2.FONT_HERSHEY_SIMPLEX,
					0.7,
					color,
					2,
				)
				cv2.imshow("Calibracion Homografia", draw)

				key = cv2.waitKey(1) & 0xFF
				if key == 27:
					print("[INFO] Calibracion cancelada por usuario")
					return None

				if key == 32 and found and refined is not None:
					object_points.append(objp.copy())
					image_points.append(refined)
					print(
						f"[INFO] Captura {len(image_points)}/{cls.REQUIRED_CAPTURES} guardada"
					)

			if not image_points or last_gray_shape is None:
				print("[ERROR] No se capturaron puntos validos")
				return None

			ret, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
				object_points,
				image_points,
				last_gray_shape,
				None,
				None,
			)
			if not ret:
				print("[ERROR] Fallo calibrateCamera")
				return None

			# Agregamos correspondencias de todas las capturas para una
			# homografia mas robusta en el plano del tablero.
			img_all = np.vstack([pts.reshape(-1, 2) for pts in image_points]).astype(
				np.float32
			)
			world_xy = objp[:, :2]
			obj_all = np.vstack([world_xy for _ in image_points]).astype(np.float32)

			homography, mask = cv2.findHomography(img_all, obj_all, method=cv2.RANSAC)
			if homography is None:
				print("[ERROR] No se pudo calcular la homografia")
				return None

			inliers = int(mask.sum()) if mask is not None else len(img_all)
			print(f"[INFO] Homografia calculada. Inliers: {inliers}/{len(img_all)}")

			out_path = cls.save_calibration(
				camera_matrix=camera_matrix,
				homography_matrix=homography,
				dist_coeffs=dist_coeffs,
				reprojection_error=float(ret),
			)

			print(f"[OK] Calibracion guardada en: {out_path}")
			return {
				"camera_matrix": camera_matrix,
				"distortion_coefficients": dist_coeffs,
				"homography_matrix": homography,
				"reprojection_error": float(ret),
			}

		finally:
			cap.release()
			cv2.destroyAllWindows()


def _print_loaded_calibration(calib_data: Dict[str, object]) -> None:
	cam = np.asarray(calib_data["camera_matrix"], dtype=np.float64)
	h = np.asarray(calib_data["homography_matrix"], dtype=np.float64)

	print("[INFO] Calibracion cargada correctamente")
	print(f"  Archivo: {HomographyCalibrator.CALIBRATION_FILE}")
	print(f"  Patron: {tuple(calib_data.get('chessboard_size', HomographyCalibrator.CHESSBOARD_SIZE))}")
	print(f"  Cuadro: {calib_data.get('square_size_mm', HomographyCalibrator.SQUARE_SIZE_MM)} mm")

	print("\nMatriz de camara:")
	print(cam)

	print("\nMatriz de homografia:")
	print(h)

	if "reprojection_error" in calib_data:
		print(f"\nError de reproyeccion: {float(calib_data['reprojection_error']):.4f}")


def _parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Calibracion por homografia usando tablero de ajedrez"
	)
	parser.add_argument(
		"-c",
		"--capture",
		action="store_true",
		help="Captura nuevas imagenes y recalcula calibracion",
	)
	parser.add_argument(
		"-l",
		"--load",
		action="store_true",
		help="Carga y muestra la calibracion guardada",
	)
	parser.add_argument(
		"--camera",
		type=int,
		default=0,
		help="Indice de camara a usar (default: 0)",
	)
	return parser.parse_args()


def main() -> int:
	args = _parse_args()

	if not args.capture and not args.load:
		print("[INFO] Debes indicar una accion: --capture o --load")
		return 1

	if args.capture:
		result = HomographyCalibrator.capture_calibration(camera_index=args.camera)
		if result is None:
			return 1

	if args.load:
		calib_data = HomographyCalibrator.load_calibration()
		if calib_data is None:
			print("[ERROR] No se encontro archivo de calibracion")
			print("        Ejecuta: python calibracion.py --capture")
			return 1
		_print_loaded_calibration(calib_data)

	return 0


if __name__ == "__main__":
	raise SystemExit(main())
