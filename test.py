import visualisation as vis
import numpy as np

data = np.loadtxt("datasets/PCPNet/Liberty100k.xyz")
norm = np.loadtxt("datasets/PCPNet/Liberty100k.normals")
x = np.concatenate((data,norm),axis=1)
vis.visPointCloudWithNormals(x)