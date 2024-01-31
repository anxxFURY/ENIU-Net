import open3d as o3d
import numpy as np

def visPointCloudWithNormals(xyz):
    pcd = o3d.geometry.PointCloud()
    num_points = len(xyz)
    color = np.zeros((num_points,3), dtype=np.float64)
	# R G B -> 
    color[:,2] = 1 # Blue
    pcd.points = o3d.utility.Vector3dVector(xyz[:,:3])
    pcd.normals = o3d.utility.Vector3dVector(xyz[:,3:])
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd],point_show_normal=True)
    
def visPointCloud(xyz):
    pcd = o3d.geometry.PointCloud()
    num_points = len(xyz)
    color = np.zeros((num_points,3), dtype=np.float64)
	# R G B -> 
    color[:,2] = 1 # Blue
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = o3d.utility.Vector3dVector(color)
    o3d.visualization.draw_geometries([pcd])
    
	