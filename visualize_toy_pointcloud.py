import numpy as np
import open3d as o3d


def sample_sphere(n_points=256):
    pts = []
    for _ in range(n_points):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(0, np.pi)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)


def sample_cube(n_points=256):
    return np.random.uniform(-1, 1, size=(n_points, 3)).astype(np.float32)


def sample_cylinder(n_points=256):
    pts = []
    for _ in range(n_points):
        theta = np.random.uniform(0, 2 * np.pi)
        r = np.random.uniform(0, 1)
        z = np.random.uniform(-1, 1)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        pts.append([x, y, z])
    return np.array(pts, dtype=np.float32)


def normalize_points(points):
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    points = points / (scale + 1e-8)
    return points.astype(np.float32)


def to_o3d_pcd(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    colors = np.tile(np.array(color, dtype=np.float64), (points.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def main():
    n_points = 256

    sphere = normalize_points(sample_sphere(n_points))
    cube = normalize_points(sample_cube(n_points))
    cylinder = normalize_points(sample_cylinder(n_points))

    # 为了方便同时看，把三个点云沿 x 轴错开
    sphere[:, 0] -= 3.0
    cylinder[:, 0] += 3.0

    sphere_pcd = to_o3d_pcd(sphere, [1.0, 0.0, 0.0])      # 红
    cube_pcd = to_o3d_pcd(cube, [0.0, 1.0, 0.0])          # 绿
    cylinder_pcd = to_o3d_pcd(cylinder, [0.0, 0.0, 1.0])  # 蓝

    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)

    o3d.visualization.draw_geometries(
        [sphere_pcd, cube_pcd, cylinder_pcd, axis],
        window_name="Toy Point Clouds: Sphere / Cube / Cylinder",
        width=1200,
        height=800
    )


if __name__ == "__main__":
    main()