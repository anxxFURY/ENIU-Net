import visualisation as vis
import numpy as np

gts = np.loadtxt("customData_GT/withNormals/1024/star_smooth100k_1024.xyz")
pred = np.loadtxt("log/res-1024/star_smooth100k_1024.xyz")
print(gts.shape)
