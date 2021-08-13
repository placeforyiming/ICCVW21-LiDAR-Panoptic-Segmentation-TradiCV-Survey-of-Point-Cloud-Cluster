
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import torch.nn.functional as Func
import argparse
from torchvision.transforms import functional as FF
from dataloader.Dataset_semanticKITTI import *
from dataloader.laserscan import SemLaserScan,LaserScan
from PC_cluster.depth_cluster.build.Depth_Cluster import Depth_Cluster 
from PC_cluster.Euclidean_cluster.build import Euclidean_Cluster
from PC_cluster.SuperVoxel_cluster.build import SuperVoxel_Cluster
from PC_cluster.ScanLineRun_cluster.build import ScanLineRun_Cluster
import random
import time
import cv2
from collections import Counter


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',dest= "dataset", default='semanticKITTI', help='')
parser.add_argument('--root',  dest= "root", default='./Dataset/semanticKITTI/',help="./Dataset/semanticKITTI/")
parser.add_argument('--range_y', dest= "range_y", default=64, help="64")
parser.add_argument('--range_x', dest= "range_x", default=2048, help="2048")
parser.add_argument('--minimum_points', dest= "minimum_points", default=40, help="minimum_points of each class")
parser.add_argument('--which_cluster', dest= "which_cluster", default=1, help="4: ScanLineRun clustering; 3: superVoxel clustering; 2: euclidean; 1: depth_cluster; ")
parser.add_argument('--mode', dest= "mode", default='val', help="val or test; ")


args = parser.parse_args()


inv_label_dict={0:0,1:10,2:11,3:15,4:18,5:20,6:30,7:31,8:32,9:40,10:44,11:48,12:49,13:50,14:51,15:70,16:71,17:72,18:80,19:81}
inv_label_dict_reverse={0:0,10:1,11:2,15:3,18:4,20:5,30:6,31:7,32:8,40:9,44:10,48:11,49:12,50:13,51:14,70:15,71:16,72:17,80:18,81:19}



def NN_filter_here(proj_range,semantic_pred,instance_pred,k_size=5):
    semantic_pred=semantic_pred.double()
    instance_pred=instance_pred.double()
    H,W=np.shape(proj_range)
    proj_range_expand=torch.unsqueeze(proj_range,axis=0)
    proj_range_expand=torch.unsqueeze(proj_range_expand,axis=0)
    semantic_pred_expand=torch.unsqueeze(semantic_pred,axis=0)
    semantic_pred_expand=torch.unsqueeze(semantic_pred_expand,axis=0)
    instance_pred_expand=torch.unsqueeze(instance_pred,axis=0)
    instance_pred_expand=torch.unsqueeze(instance_pred_expand,axis=0)
    pad = int((k_size - 1) / 2)
    proj_unfold_range = Func.unfold(proj_range_expand,kernel_size=(k_size, k_size),padding=(pad, pad))
    proj_unfold_range = proj_unfold_range.reshape(-1, k_size*k_size, H, W)
        
    proj_unfold_pre_sem = Func.unfold(semantic_pred_expand,kernel_size=(k_size, k_size),padding=(pad, pad))
    proj_unfold_pre_sem = proj_unfold_pre_sem.reshape(-1, k_size*k_size, H, W)

    proj_unfold_pre_ins = Func.unfold(instance_pred_expand,kernel_size=(k_size, k_size),padding=(pad, pad))
    proj_unfold_pre_ins = proj_unfold_pre_ins.reshape(-1, k_size*k_size, H, W)

    return proj_unfold_range,proj_unfold_pre_sem,proj_unfold_pre_ins



if args.which_cluster==1:
	cluster=Depth_Cluster(0.15,9)
if args.which_cluster==2:
	cluster=Euclidean_Cluster.Euclidean_Cluster(0.5,int(args.minimum_points),10000)
if args.which_cluster==3:
	cluster=SuperVoxel_Cluster.SuperVoxel_Cluster(0.5, 8, 0.0, 1.0, 0.0)
if args.which_cluster==4:
	cluster=ScanLineRun_Cluster.ScanLineRun_Cluster(0.5, 1)

CFG = yaml.safe_load(open(args.root+'semantic-kitti.yaml', 'r'))
label_transfer_dict =CFG["learning_map"]


	

for zzz in range(11,22):
	A=LaserScan(project=True, flip_sign=False, H=args.range_y, W=args.range_x, fov_up=3.0, fov_down=-25.0)
	
	if args.mode=='test':
		lidar_list=glob.glob(args.root+'/data_odometry_velodyne/*/*/'+'test/'+str(zzz)+'/*/*.bin')
			   

		if not os.path.exists("./method_predictions/"):
			print ("inference semantic first")
		if not os.path.exists("./method_predictions/sequences/"):
			print ("inference semantic first")


		save_path_for_prediction="./method_predictions/sequences/"+str(zzz)+"/"
		if not os.path.exists(save_path_for_prediction):
			print ("inference semantic first")

		save_path_for_prediction="./method_predictions/sequences/"+str(zzz)+"/predictions/"
		if not os.path.exists(save_path_for_prediction):
			print ("inference semantic first")


	if args.mode=='val':
		lidar_list=glob.glob(args.root+'/data_odometry_velodyne/*/*/'+'val'+'/*/*/*.bin')
			   
		if not os.path.exists("./method_predictions/"):
			print ("inference semantic first")
		if not os.path.exists("./method_predictions/sequences/"):
			print ("inference semantic first")


		save_path_for_prediction="./method_predictions/sequences/08/"
		if not os.path.exists(save_path_for_prediction):
			print ("inference semantic first")


		save_path_for_prediction="./method_predictions/sequences/08/predictions/"
		if not os.path.exists(save_path_for_prediction):
			print ("inference semantic first")

		print (len(lidar_list))
	if args.mode=='val' and zzz>11:
		continue

	time_list=[]
	for i in range(len(lidar_list)):


		if i%100==0:
			print (i)
		path_list=lidar_list[i].split('/')
		label_file=save_path_for_prediction+path_list[-1][:len(path_list[-1])-3]+"label"


		A.open_scan(lidar_list[i])

		print (lidar_list[i])
		semantic_label=np.fromfile(label_file,dtype=np.uint32)

		semantic_label = semantic_label.reshape((-1))

		semantic_label = semantic_label & 0xFFFF

		semantic_label_inv=[inv_label_dict_reverse[mm] for mm in semantic_label]

		label_img=np.zeros((64,2048))
		depth_img=np.zeros((64,2048))
		covered_points=[]
		for jj in range(len(A.proj_x)):
			y_range,x_range=A.proj_y[jj],A.proj_x[jj]
			if label_img[y_range,x_range]==0:
				label_img[y_range,x_range]=semantic_label_inv[jj]
				depth_img[y_range,x_range]=A.unproj_range[jj]


	
		if args.which_cluster==1:
			mask=(label_img<9)
			range_img_pre=A.proj_range*mask
			range_img=range_img_pre.reshape(-1)
			a=time.time()
			instance_label=cluster.Depth_cluster(range_img)
			b=time.time()
			time_list.append(b-a)
			instance_label=np.asarray(instance_label).reshape(64,2048)


		if args.which_cluster==2:
			mask=np.logical_and(label_img>0,label_img<9)
			range_img_pre_x=A.proj_xyz[:,:,0]*mask
			range_img_pre_y=A.proj_xyz[:,:,1]*mask
			range_img_pre_z=A.proj_xyz[:,:,2]*mask
			# process the voxelized point cloud to save comlexity of kd-tree
			voxel_size=0.1
			range_img_pre_x_index=np.round(range_img_pre_x/voxel_size)
			range_img_pre_y_index=np.round(range_img_pre_y/voxel_size)
			range_img_pre_z_index=np.round(range_img_pre_z/voxel_size)
			exist_index=[]
			mm_list=[]
			nn_list=[]
			index_mask=np.zeros((64,2048))
			for m in range(64):
				for n in range(2048):
					each_index=str(range_img_pre_x_index[m,n])+'_'+str(range_img_pre_y_index[m,n])+'_'+str(range_img_pre_z_index[m,n])
					if each_index in exist_index:
						continue
					else:
						exist_index.append(each_index)
						mm_list.append(m)
						nn_list.append(n)
						index_mask[m,n]=1
			range_img_pre_x=A.proj_xyz[:,:,0]*index_mask
			range_img_pre_y=A.proj_xyz[:,:,1]*index_mask
			range_img_pre_z=A.proj_xyz[:,:,2]*index_mask
			range_img_x=range_img_pre_x.reshape(-1)
			range_img_y=range_img_pre_y.reshape(-1)
			range_img_z=range_img_pre_z.reshape(-1)
			mask_a=index_mask.reshape(-1)
			total_points=np.sum(mask_a).astype(int)
			#print (total_points)

			a=time.time()
			instance_label=cluster.Euclidean_cluster(range_img_x,range_img_y,range_img_z,mask_a,total_points)
			b=time.time()
			time_list.append(b-a)
			instance_label=np.asarray(instance_label).reshape(64,2048)


			look_up_dict={}
			for mm in range(len(exist_index)):
				look_up_dict[exist_index[mm]]=instance_label[mm_list[mm],nn_list[mm]]
			for m in range(64):
				for n in range(2048):
					each_index=str(range_img_pre_x_index[m,n])+'_'+str(range_img_pre_y_index[m,n])+'_'+str(range_img_pre_z_index[m,n])
					instance_label[m,n]=look_up_dict[each_index]
			instance_label=instance_label*mask
		

		# Supervoxel Clustering
		if args.which_cluster==3:
			mask=np.logical_and(label_img>0,label_img<9)
			range_img_x=A.proj_xyz[:,:,0]*mask
			range_img_y=A.proj_xyz[:,:,1]*mask
			range_img_z=A.proj_xyz[:,:,2]*mask

			# print ('input cloud size', range_img_x.shape)
			
			width = 64
			height = 2048

			# print(range_img_x)
			a=time.time()		
			instance_label=cluster.SuperVoxel_cluster(range_img_x,range_img_y,range_img_z,width,height)
			b=time.time()
			time_list.append(b-a)
			assert(len(instance_label)==64 and len(instance_label[0])==2048)
			instance_label=np.array(instance_label)
			instance_label=instance_label*mask

			# print(instance_label)

		# ScanLineRun Clustering
		if args.which_cluster==4:
			mask=np.logical_and(label_img>0,label_img<9)
			range_img_x=A.proj_xyz[:,:,0]*mask
			range_img_y=A.proj_xyz[:,:,1]*mask
			range_img_z=A.proj_xyz[:,:,2]*mask

			width = 64
			height = 2048
			a=time.time()
			instance_label=cluster.ScanLineRun_cluster(range_img_x,range_img_y,range_img_z,mask,width,height)
			b=time.time()
			time_list.append(b-a)
			instance_label=np.array(instance_label)

		
		print (np.sum(time_list)/len(time_list))
		true_lable=0
		for mm in np.unique(instance_label):
			if np.sum(mm==instance_label)>args.minimum_points:
				true_lable+=1
			else:
				instance_label[mm==instance_label]=0



		color_rgb=np.zeros((64,2048,3))
		all_instance_lables= np.unique(instance_label)
		for i in all_instance_lables:
			if i>0:

				temp_labels = Counter(label_img[instance_label==i])
				temp_dict={5:0}
				if np.min(temp_labels.keys())>temp_dict.keys():
					changed_label=temp_labels.most_common(1)[0][0]
					label_img[np.where(instance_label==i)]=changed_label

				rgb_x=random.randint(0,254)
				rgb_y=random.randint(0,254)
				rgb_z=random.randint(0,254)
				temp_mask=instance_label==i
				color_rgb[:,:,0]+=temp_mask*rgb_x
				color_rgb[:,:,1]+=temp_mask*rgb_y
				color_rgb[:,:,2]+=temp_mask*rgb_z




		plt.imsave('./output_example.png',np.asarray(color_rgb).astype(np.uint8))
		#time.sleep(1)

		
		t_1=torch.squeeze(torch.from_numpy(depth_img))
		t_2=torch.squeeze(torch.from_numpy(label_img))
		t_3=torch.squeeze(torch.from_numpy(instance_label))
		proj_unfold_range,proj_unfold_sem,proj_unfold_ins=NN_filter_here(t_1,t_2,t_3)
		proj_unfold_range=proj_unfold_range.cpu().numpy()
		proj_unfold_sem=proj_unfold_sem.cpu().numpy()
		proj_unfold_ins=proj_unfold_ins.cpu().numpy()

		label=[]
		for jj in range(len(A.proj_x)):
			y_range,x_range=A.proj_y[jj],A.proj_x[jj]
			if A.unproj_range[jj]==depth_img[y_range,x_range]:
				semantic_label_each=label_img[y_range,x_range] 
				instance_label_each=instance_label[y_range,x_range]
				lower_half=inv_label_dict[semantic_label_each]
				upper_half=instance_label_each.astype(np.long)
				label_each = (upper_half << 16) + lower_half
				label.append(label_each)
			else:
				if semantic_label_inv[jj]<9:
					potential_label_sem=proj_unfold_sem[0,:,y_range,x_range]
					potential_label_ins=proj_unfold_ins[0,:,y_range,x_range]
					potential_range=proj_unfold_range[0,:,y_range,x_range]
					min_arg=np.argmin(abs(potential_range-A.unproj_range[jj]))
					lower_half=inv_label_dict[potential_label_sem[min_arg]]
					upper_half=potential_label_ins[min_arg].astype(np.long)
					label_each = (upper_half << 16) + lower_half
					label.append(label_each)
				else:
					semantic_label_each=semantic_label_inv[jj]
					instance_label_each=0
					lower_half=inv_label_dict[semantic_label_each]
					upper_half=instance_label_each
					label_each = (upper_half << 16) + lower_half
					label.append(label_each)
		label=np.asarray(label)
		label = label.astype(np.uint32)
		label.tofile(label_file)
		