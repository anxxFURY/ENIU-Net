import numpy as np
import visualisation
from numpy.linalg import norm
import open3d as o3d
from matplotlib import cm


def generate_heat_map(point_cloud, cosine_similarity):
    # Define color map
    cmap = cm.get_cmap("hsv_r")
    colors = cmap(cosine_similarity.flatten())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        point_cloud[:, :3]
    )  # assuming only XYZ coordinates
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    o3d.visualization.draw_geometries([pcd])


def cosine_sim(normal_gts, pred_normals, save_pth=None):
    """
    Calculate the cosine similarity between pairs of vectors.

    Parameters:
    - normal_gt (numpy array): Array containing ground truth normal vectors (assumed to be row vectors).
    - pred_normal (numpy array): Array containing predicted normal vectors (assumed to be row vectors).

    Returns:
    - cosine (numpy array): Array of cosine similarity values for each pair of vectors.

    The cosine similarity between two vectors A and B is calculated as the dot product of A and B
    divided by the product of their magnitudes. The function operates along axis 1, assuming each row
    represents a vector. The result is an array of cosine similarity values for each pair of vectors.
    """
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pred_normals)
    # pcd.estimate_normals()

    # pred_normals = np.asarray(pcd.normals)

    cosine = np.sum(normal_gts * pred_normals, axis=1) / (
        norm(normal_gts, axis=1) * norm(pred_normals, axis=1)
    )
    if save_pth is not None:
        np.savetxt(save_pth, cosine, fmt="%.2f")

    return cosine


def normal_RMSE(normal_gts, normal_preds, eval_file):
    """
    Compute normal root-mean-square error (RMSE)
    normal_gts = list of Ground truth normals
    normal_preds = list of prediction normals
    """
  
    def l2_norm(v):
        norm_v = np.sqrt(np.sum(np.square(v), axis=1))
        return norm_v

    log_file = open(eval_file, "w")

    def log_string(out_str):
        log_file.write(out_str + "\n")
        log_file.flush()
        # print(out_str)

    rms = []
    rms_o = []
    pgp30 = []
    pgp25 = []
    pgp20 = []
    pgp15 = []
    pgp10 = []
    pgp5 = []
    pgp_alpha = []

    for i in range(len(normal_gts)):
        normal_gt = normal_gts[i]
        normal_pred = normal_preds[i]

        normal_gt_norm = l2_norm(normal_gt)
        normal_results_norm = l2_norm(normal_pred)
        normal_pred = np.divide(
            normal_pred, np.tile(np.expand_dims(
                normal_results_norm, axis=1), [1, 3])
        )
        normal_gt = np.divide(
            normal_gt, np.tile(np.expand_dims(normal_gt_norm, axis=1), [1, 3])
        )

        # Unoriented RMSE
        ####################################################################
        nn = np.sum(np.multiply(normal_gt, normal_pred), axis=1)
        nn[nn > 1] = 1
        nn[nn < -1] = -1

        ang = np.rad2deg(np.arccos(np.abs(nn)))

        # portion of good points
        rms.append(np.sqrt(np.mean(np.square(ang))))
        pgp30_shape = sum([j < 30.0 for j in ang]) / float(len(ang))
        pgp25_shape = sum([j < 25.0 for j in ang]) / float(len(ang))
        pgp20_shape = sum([j < 20.0 for j in ang]) / float(len(ang))
        pgp15_shape = sum([j < 15.0 for j in ang]) / float(len(ang))
        pgp10_shape = sum([j < 10.0 for j in ang]) / float(len(ang))
        pgp5_shape = sum([j < 5.0 for j in ang]) / float(len(ang))
        pgp30.append(pgp30_shape)
        pgp25.append(pgp25_shape)
        pgp20.append(pgp20_shape)
        pgp15.append(pgp15_shape)
        pgp10.append(pgp10_shape)
        pgp5.append(pgp5_shape)

        pgp_alpha_shape = []
        for alpha in range(30):
            pgp_alpha_shape.append(
                sum([j < alpha for j in ang]) / float(len(ang)))
        pgp_alpha.append(pgp_alpha_shape)

        # Oriented RMSE
        ####################################################################
        ang_o = np.rad2deg(np.arccos(nn))  # angle error in degree
        ids = ang_o > 90.0
        p = sum(ids) / normal_pred.shape[0]

        # if more than half of points have wrong orientation, then flip all normals
        # if p > 0.5:
        # 	nn = np.sum(np.multiply(normal_gt, -1 * normal_pred), axis=1)
        # 	nn[nn > 1] = 1
        # 	nn[nn < -1] = -1
        # 	ang_o = np.rad2deg(np.arccos(nn))    # angle error in degree
        # 	ids = ang_o > 90.0
        # 	p = sum(ids) / normal_pred.shape[0]

        rms_o.append(np.sqrt(np.mean(np.square(ang_o))))

    avg_rms = np.mean(rms)
    avg_rms_o = np.mean(rms_o)
    avg_pgp30 = np.mean(pgp30)
    avg_pgp25 = np.mean(pgp25)
    avg_pgp20 = np.mean(pgp20)
    avg_pgp15 = np.mean(pgp15)
    avg_pgp10 = np.mean(pgp10)
    avg_pgp5 = np.mean(pgp5)
    avg_pgp_alpha = np.mean(np.array(pgp_alpha), axis=0)

    log_string("RMS per shape: " + str(rms))
    log_string("RMS not oriented (shape average): " + str(avg_rms))
    log_string("RMS oriented (shape average): " + str(avg_rms_o))
    log_string("PGP30 per shape: " + str(pgp30))
    log_string("PGP25 per shape: " + str(pgp25))
    log_string("PGP20 per shape: " + str(pgp20))
    log_string("PGP15 per shape: " + str(pgp15))
    log_string("PGP10 per shape: " + str(pgp10))
    log_string("PGP5 per shape: " + str(pgp5))
    log_string("PGP30 average: " + str(avg_pgp30))
    log_string("PGP25 average: " + str(avg_pgp25))
    log_string("PGP20 average: " + str(avg_pgp20))
    log_string("PGP15 average: " + str(avg_pgp15))
    log_string("PGP10 average: " + str(avg_pgp10))
    log_string("PGP5 average: " + str(avg_pgp5))
    log_string("PGP alpha average: " + str(avg_pgp_alpha))
    log_file.close()

    return avg_rms, avg_rms_o


def read_shapes_list(file_path):
    with open(file_path, "r") as file:
        shapes_list = file.read().splitlines()
    return shapes_list


if __name__ == "__main__":
    shape_list = read_shapes_list("testset_all_4096.txt")

    gt_normals = []
    pred_normals = []

    # Change accordingly...
    for shape in shape_list:
        input_point_cloud = np.loadtxt(f"customDataNew/withoutNormals/4096/{shape}.xyz")
        gt_normal = np.loadtxt(f"customDataNew/onlyNormals/4096/{shape}.normals")
        pred_normal = np.loadtxt(f"/Users/aniruddhashadagali/All-Codes/PythonCode/Experiments/Dataset/PCPNet/log_ours_4096/001/results_PCPNet/ckpt_800/pred_normal/{shape}.normals")

        visualisation.visPointCloudWithHeatMap(input_point_cloud, cosine_sim(gt_normal, pred_normal), f"visualisation_Results/ours/4096/{shape}.ply", False)

        pred_normals.append(pred_normal)
        gt_normals.append(gt_normal)
    
    _,_ = normal_RMSE(gt_normals, pred_normals, f"rmse_Results/ours_4096.txt")
    # o = np.loadtxt("log/res-SHS-4096/star_smooth100k_4096.xyz")
    # visualisation.visPointCloudWithHeatMap(
    #     pc[:, :3], cosine_sim(pc[:, 3:], o[:,3:]))

