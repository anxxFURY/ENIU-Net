import numpy as np
import os
import open3d as o3d
import visualisation

def farthest_point_sample(point, npoint):
	"""
	Input:
		xyz: pointcloud data, [N, D]
		label : per point label ,[N]
		npoint: number of samples
	Return:
		centroids: sampled pointcloud index, [npoint, D]
	"""
	N, D = point.shape
	xyz = point[:, :3]
	centroids = np.zeros((npoint,))
	distance = np.ones((N,)) * 1e10
	farthest = np.random.randint(0, N)
	for i in range(npoint):
		centroids[i] = farthest
		centroid = xyz[farthest, :]
		dist = np.sum((xyz - centroid) ** 2, -1)
		mask = dist < distance
		distance[mask] = dist[mask]
		farthest = np.argmax(distance, -1)
	point = point[centroids.astype(np.int32)]
	# label = label[centroids.astype(np.int32)]
	return point  # ,label

def loadpc(file_list):

	for file in file_list:
		pc = np.loadtxt('datasets/PCPNet/' + file + '.xyz')
		normals = np.loadtxt('datasets/PCPNet/' + file + '.normals')

		pc_with_normals = np.concatenate((pc,normals),axis=1)
		ratios = [1024,2048,4096,8192]

		for r in ratios:
			
			pc_with_normals_fps = farthest_point_sample(pc_with_normals,r)
			if not os.path.exists(f"customData/withNormals/{r}"):
				os.makedirs(f"customData/withNormals/{r}")
			np.savetxt(f"customData/withNormals/{r}/{file}_{r}.xyz",pc_with_normals_fps)

			if not os.path.exists(f"customData/onlyNormals/{r}"):
				os.makedirs(f"customData/onlyNormals/{r}")
			np.savetxt(f"customData/onlyNormals/{r}/{file}_{r}.normals",pc_with_normals_fps[:,3:])
			
			if not os.path.exists(f"customData/withoutNormals/{r}"):
				os.makedirs(f"customData/withoutNormals/{r}")
			np.savetxt(f"customData/withoutNormals/{r}/{file}_{r}.xyz",pc_with_normals_fps[:,:3])



if __name__ == "__main__":
	file_list_filename = 'datasets/PCPNet/list/mytest.txt'
	file_list = [line.strip() for line in open(file_list_filename)]
	loadpc(file_list)