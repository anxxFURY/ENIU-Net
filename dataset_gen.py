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
        pc = np.loadtxt("/Users/aniruddhashadagali/Downloads/PCPNet/" + file + ".xyz")
        normals = np.loadtxt("/Users/aniruddhashadagali/Downloads/PCPNet/" + file + ".normals")

        pc_with_normals = np.concatenate((pc, normals), axis=1)
        ratios = [1024, 4096]

        for r in ratios:
            pc_with_normals_fps = farthest_point_sample(pc_with_normals, r)
            if not os.path.exists(f"customDataNew/withNormals/{r}"):
                os.makedirs(f"customDataNew/withNormals/{r}")
            np.savetxt(
                f"customDataNew/withNormals/{r}/{file}_{r}.xyz", pc_with_normals_fps
            )

            if not os.path.exists(f"customDataNew/onlyNormals/{r}"):
                os.makedirs(f"customDataNew/onlyNormals/{r}")
            np.savetxt(
                f"customDataNew/onlyNormals/{r}/{file}_{r}.normals",
                pc_with_normals_fps[:, 3:],
            )

            if not os.path.exists(f"customDataNew/withoutNormals/{r}"):
                os.makedirs(f"customDataNew/withoutNormals/{r}")
            np.savetxt(
                f"customDataNew/withoutNormals/{r}/{file}_{r}.xyz",
                pc_with_normals_fps[:, :3],
            )


def con_pc_with_up(file_list):
    for file in file_list:
        pc = np.loadtxt("customDataNew/withoutNormals/try/" + file + "_1024.xyz")
        pc_up = np.loadtxt("upsampled/" + file + "_1024.xyz")

        res = np.concatenate((pc, pc_up), axis=0)

        np.savetxt(f"datasets_5120/PCPNet/{file}_5120.xyz", res)


if __name__ == "__main__":
    file_list_filename = "/Users/aniruddhashadagali/Downloads/PCPNet/list/testset_all.txt"
    file_list = [line.strip() for line in open(file_list_filename)]

    # loadpc(file_list=file_list)
    con_pc_with_up(file_list=file_list)
    # for file in file_list:
    #     path = f"customDataNew_GT/withNormals/4096/{file}_4096.xyz"
    #     pred = f"/Users/aniruddhashadagali/Downloads/pred_normal/{file}_4096.normals"

    #     pc = np.loadtxt(path)
    #     pred_pc = np.loadtxt(pred)
    #     res = np.concatenate((pc[:, :3], pred_pc), axis=1)
    #     np.savetxt(f"log/res-SHS-4096/{file}_4096.xyz",res)
