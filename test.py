import visualisation as vis
import numpy as np

def gen_res(file_list):
	for file in file_list:
		pc_5120  = np.loadtxt('datasets_2/PCPNet/' + file + '.xyz')
		norm_pred_5120 = np.loadtxt('results_PCPNet_for_5120/ckpt_800/pred_normal/' + file + '.normals')
		print(pc_5120.shape)
		print(norm_pred_5120.shape)
		res_5120 = np.concatenate((pc_5120,norm_pred_5120),axis=1)
		np.savetxt(f"log/res-5120/{file}_5120.xyz",res_5120)

		get_1024 = res_5120[:1024,:]
		np.savetxt(f"log/res-1024/{file}_1024.xyz",get_1024)

file_list_filename = 'hey.txt'
file_list = [line.strip() for line in open(file_list_filename)]
gen_res(file_list)