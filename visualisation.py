import open3d as o3d
import numpy as np
from matplotlib import cm
from PIL import Image


def visPointCloudWithNormals(xyz):
    pcd = o3d.geometry.PointCloud()
    num_points = len(xyz)
    color = np.zeros((num_points, 3), dtype=np.float64)
    # R G B ->
    #R color[:, 0]
    #G color[:, 1]
    #B color[:,2]
    
    color[:, 2] = 1  # Blue

    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(xyz[:, 3:])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)


def visPointCloud(xyz, blue=True):
    pcd = o3d.geometry.PointCloud()
    num_points = len(xyz)
    color = np.zeros((num_points, 3), dtype=np.float64)
    # R G B ->
    if blue:
        color[:, 2] = 1  # Blue
    else:
        color[:] = [0.54117647, 0.16862745, 0.88627451]
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud("paper_results/liberty/1.ply", pcd)
    o3d.visualization.draw_geometries([pcd])


def visPointCloudWithHeatMap(xyz, cosine):
    pcd = o3d.geometry.PointCloud()
    num_points = len(xyz)
    color = np.zeros((num_points, 3), dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])

    for i in range(len(cosine)):
        # voilet ->
        if cosine[i] <= 1 and cosine[i] > 0.75:
            color[i] = [0.54117647, 0.16862745, 0.88627451]
        elif cosine[i] <= 0.75 and cosine[i] > 0.5:
            color[i] = np.array([0, 0, 255]) / 255.0
        elif cosine[i] <= 0.5 and cosine[i] > 0.25:
            color[i] = np.array([0, 255, 0]) / 255.0
        elif cosine[i] <= 0.25 and cosine[i] > 0:
            color[i] = np.array([255, 255, 0]) / 255.0
        elif cosine[i] <= 0 and cosine[i] > (-0.5):
            color[i] = np.array([255, 165, 0]) / 255.0
        else:
            color[i] = np.array([255, 0, 0]) / 255.0

    pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.io.write_point_cloud("paper_results/liberty/1.ply", pcd)
    o3d.visualization.draw_geometries([pcd])


def meshingPointCloud(xyz, file_path="test.obj"):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(xyz[:, 3:])
    print("run Poisson surface reconstruction")
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
    o3d.io.write_triangle_mesh(file_path, mesh)
    o3d.visualization.draw_geometries([mesh])


def reconstruction(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    radii = [0.005, 0.01, 0.02, 0.04]
    # pcd.normals = o3d.utility.Vector3dVector(np.zeros((1, 3)))
    pcd.estimate_normals()
    rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    o3d.visualization.draw_geometries([pcd, rec_mesh])
    o3d.io.write_triangle_mesh("test.obj", rec_mesh)
