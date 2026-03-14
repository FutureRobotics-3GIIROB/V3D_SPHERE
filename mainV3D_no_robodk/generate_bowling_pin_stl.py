from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


Vec3 = Tuple[float, float, float]


def normalize(v: Vec3) -> Vec3:
    """Return normalized vector or zero vector for degenerate input."""
    x, y, z = v
    n = math.sqrt(x * x + y * y + z * z)
    if n <= 1e-12:
        return (0.0, 0.0, 0.0)
    return (x / n, y / n, z / n)


def sub(a: Vec3, b: Vec3) -> Vec3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def cross(a: Vec3, b: Vec3) -> Vec3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def triangle_normal(v1: Vec3, v2: Vec3, v3: Vec3) -> Vec3:
    """Compute a unit normal for one triangle."""
    return normalize(cross(sub(v2, v1), sub(v3, v1)))


def ring_vertices(radius: float, z: float, segments: int) -> List[Vec3]:
    """Create one horizontal ring of vertices."""
    verts: List[Vec3] = []
    for i in range(segments):
        theta = 2.0 * math.pi * i / segments
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        verts.append((x, y, z))
    return verts


def write_ascii_stl(path: Path, name: str, facets: Iterable[Tuple[Vec3, Vec3, Vec3]]) -> None:
    """Write facets to an ASCII STL file."""
    lines: List[str] = [f"solid {name}"]
    for v1, v2, v3 in facets:
        n = triangle_normal(v1, v2, v3)
        lines.append(f"  facet normal {n[0]:.8e} {n[1]:.8e} {n[2]:.8e}")
        lines.append("    outer loop")
        lines.append(f"      vertex {v1[0]:.8e} {v1[1]:.8e} {v1[2]:.8e}")
        lines.append(f"      vertex {v2[0]:.8e} {v2[1]:.8e} {v2[2]:.8e}")
        lines.append(f"      vertex {v3[0]:.8e} {v3[1]:.8e} {v3[2]:.8e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_bowling_pin_facets(segments: int = 72) -> List[Tuple[Vec3, Vec3, Vec3]]:
    """Generate a bowling-pin-like mesh using a lathe profile."""
    # Profile points: (z in meters, radius in meters), bottom -> top.
    profile: Sequence[Tuple[float, float]] = (
        (0.000, 0.022),
        (0.012, 0.029),
        (0.030, 0.021),
        (0.055, 0.015),
        (0.085, 0.018),
        (0.115, 0.023),
        (0.138, 0.016),
        (0.158, 0.007),
        (0.168, 0.003),
    )

    rings = [ring_vertices(r, z, segments) for z, r in profile]
    facets: List[Tuple[Vec3, Vec3, Vec3]] = []

    # Side surface.
    for ring_idx in range(len(rings) - 1):
        lower = rings[ring_idx]
        upper = rings[ring_idx + 1]
        for i in range(segments):
            j = (i + 1) % segments
            v00 = lower[i]
            v01 = lower[j]
            v10 = upper[i]
            v11 = upper[j]
            facets.append((v00, v10, v11))
            facets.append((v00, v11, v01))

    # Bottom cap.
    bottom_center: Vec3 = (0.0, 0.0, profile[0][0])
    bottom_ring = rings[0]
    for i in range(segments):
        j = (i + 1) % segments
        facets.append((bottom_center, bottom_ring[j], bottom_ring[i]))

    # Top cap.
    top_center: Vec3 = (0.0, 0.0, profile[-1][0])
    top_ring = rings[-1]
    for i in range(segments):
        j = (i + 1) % segments
        facets.append((top_center, top_ring[i], top_ring[j]))

    return facets


def main() -> int:
    """Generate an STL bowling pin in the current module folder."""
    out_dir = Path(__file__).resolve().parent
    out_file = out_dir / "bolo.stl"

    facets = generate_bowling_pin_facets(segments=72)
    write_ascii_stl(out_file, name="bolo", facets=facets)

    print(f"Generated STL: {out_file}")
    print(f"Triangles: {len(facets)}")
    print("Units: meters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
