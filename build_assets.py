"""Builds the seven YCB waste items used by waste_env.py into assets/ycb/ (same code as the Part C v6 notebook).

    python build_assets.py

For each object: downloads the 16k-face scan from the YCB object set (Calli et al., 2015; CC BY 4.0), builds the
collision shape (PyBullet V-HACD convex decomposition; one 80-vertex convex hull for the cups, which are only grasped
from the outside), and writes a URDF with the YCB mass and the centre of mass at the centre of the bounding box.
Existing objects are kept. Then prints SHA-1 checksums of the files the physics uses, to compare with the reference.
"""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull

ASSET_DIR = Path(__file__).resolve().parent / "assets" / "ycb"
# name: (category, mass in kg from the YCB set, collision shape)
YCB_OBJECTS = {
    "006_mustard_bottle": ("plastic", 0.603, "vhacd"), "065-a_cups": ("plastic", 0.013, "hull"),
    "065-c_cups": ("plastic", 0.017, "hull"), "004_sugar_box": ("paper", 0.514, "vhacd"),
    "009_gelatin_box": ("paper", 0.097, "vhacd"), "005_tomato_soup_can": ("metal", 0.349, "vhacd"),
    "010_potted_meat_can": ("metal", 0.370, "vhacd"),
}
YCB_URL = "http://ycb-benchmarks.s3-website-us-east-1.amazonaws.com/data/google/{}_google_16k.tgz"
URDF_TEMPLATE = """<?xml version="1.0"?>
<robot name="{name}">
  <link name="baseLink">
    <contact><lateral_friction value="0.8"/><rolling_friction value="0.001"/><spinning_friction value="0.001"/></contact>
    <inertial>
      <origin rpy="0 0 0" xyz="{cx:.5f} {cy:.5f} {cz:.5f}"/>
      <mass value="{mass}"/>
      <inertia ixx="1e-4" ixy="0" ixz="0" iyy="1e-4" iyz="0" izz="1e-4"/>
    </inertial>
    <visual><geometry><mesh filename="textured.obj" scale="1 1 1"/></geometry></visual>
    <collision><geometry><mesh filename="collision.obj" scale="1 1 1"/></geometry></collision>
  </link>
</robot>
"""


def write_convex_hull(vertices, path, n_points=80):
    # One convex hull, reduced to n_points vertices by farthest-point sampling (keeps the outer shape).
    hull_points = vertices[ConvexHull(vertices).vertices]
    chosen = [int(np.argmax(np.linalg.norm(hull_points - hull_points.mean(0), axis=1)))]
    distance = np.linalg.norm(hull_points - hull_points[chosen[0]], axis=1)
    while len(chosen) < n_points:
        chosen.append(int(np.argmax(distance)))
        distance = np.minimum(distance, np.linalg.norm(hull_points - hull_points[chosen[-1]], axis=1))
    points = hull_points[chosen]
    hull = ConvexHull(points)
    index = {old: new for new, old in enumerate(hull.vertices)}
    with open(path, "w") as f:
        f.write("o convex_hull\n")
        for i in hull.vertices:
            f.write("v %.6f %.6f %.6f\n" % tuple(points[i]))
        for simplex in hull.simplices:
            a, b, c = points[simplex]
            if np.dot(np.cross(b - a, c - a), a - points.mean(0)) < 0:   # outward-facing triangles
                simplex = simplex[[0, 2, 1]]
            f.write("f %d %d %d\n" % tuple(index[i] + 1 for i in simplex))


def pybullet_client():
    import pybullet
    if not pybullet.isConnected():
        pybullet.connect(pybullet.DIRECT)
    return pybullet


def build_ycb_asset(name, mass, collision):
    folder = ASSET_DIR / name
    if (folder / "model.urdf").exists():
        return
    folder.mkdir(parents=True, exist_ok=True)
    archive = folder / "download.tgz"
    urllib.request.urlretrieve(YCB_URL.format(name), archive)
    with tarfile.open(archive) as tar:
        for member in ["textured.obj", "textured.mtl", "texture_map.png"]:
            source = tar.extractfile(f"{name}/google_16k/{member}")
            (folder / member).write_bytes(source.read())
    archive.unlink()
    vertices = np.array([[float(x) for x in line.split()[1:4]]
                         for line in open(folder / "textured.obj") if line.startswith("v ")])
    if collision == "hull":
        write_convex_hull(vertices, folder / "collision.obj")
    else:
        pybullet_client().vhacd(str(folder / "textured.obj"), str(folder / "collision.obj"), str(folder / "vhacd.log"),
                                resolution=200000)
    centre = (vertices.min(0) + vertices.max(0)) / 2
    (folder / "model.urdf").write_text(URDF_TEMPLATE.format(name=name, cx=centre[0], cy=centre[1], cz=centre[2], mass=mass))


def checksums():
    files = [ASSET_DIR / "catalogue.json"] + [ASSET_DIR / n / f for n in YCB_OBJECTS for f in ("collision.obj", "model.urdf")]
    return {str(f.relative_to(ASSET_DIR)): hashlib.sha1(f.read_bytes()).hexdigest()[:12] for f in files}


if __name__ == "__main__":
    catalogue = {}
    for name, (category, mass, collision) in YCB_OBJECTS.items():
        build_ycb_asset(name, mass, collision)
        vertices = np.array([[float(x) for x in line.split()[1:4]]
                             for line in open(ASSET_DIR / name / "textured.obj") if line.startswith("v ")])
        size = (vertices.max(0) - vertices.min(0)).round(4).tolist()
        catalogue[name] = {"category": category, "mass_kg": mass, "size_m": size}
        print(f"{name:<22} {category:<8} {mass:5.3f} kg   size x, y, z = {', '.join(f'{100 * s:.1f}' for s in size)} cm")
    with open(ASSET_DIR / "catalogue.json", "w") as f:
        json.dump(catalogue, f, indent=2)
    print(json.dumps(checksums(), indent=1))
