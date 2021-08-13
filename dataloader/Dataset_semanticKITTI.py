import os
import numpy as np
from PIL import Image, ImageOps
import glob
import torch
from torch.utils import data
from torchvision.transforms import functional as F
import random
import matplotlib.pyplot as plt
import yaml
import pickle
from .laserscan import SemLaserScan,LaserScan
#rom laserscan import SemLaserScan




class Dataset_semanticKITTI(data.Dataset):
	

	def __init__(self,
				 root="./Dataset/semanticKITTI/",
				 split="train",
				 is_train=True,
				 range_img_size=(128, 2048),
				 if_aug=True,
				 if_range_mask=True,
				 if_perturb=True,
				 if_remission=True,
				 if_range=True,
				 flip_sign=True

				 ):

		# root= ./Dataset/semanticKITTI/
		# split= 'train' or 'val' or 'trainval'
		self.root = root
		self.split = split
		self.is_train = is_train

		self.range_h, self.range_w = range_img_size

		self.if_aug = if_aug
		self.if_range_mask=if_range_mask
		self.if_perturb=if_perturb
		self.if_remission=if_remission
		self.if_range=if_range,
		self.flip_sign=flip_sign
		

		self.CFG = yaml.safe_load(open(root+'semantic-kitti.yaml', 'r'))
		
		self.color_dict = self.CFG["color_map"]

		self.label_transfer_dict =self.CFG["learning_map"]

		self.nclasses = len(self.color_dict)

		self.A=SemLaserScan(nclasses=self.nclasses , sem_color_dict=self.color_dict, project=True, flip_sign=self.flip_sign, H=self.range_h, W=self.range_w, fov_up=3.0, fov_down=-25.0)


		if self.split=='train' or self.split=='val': 
			self.lidar_list=glob.glob(root+'/data_odometry_velodyne/*/*/'+self.split+'/*/*/*.bin')
		if self.split=='trainval':
			self.lidar_list=glob.glob(root+'/data_odometry_velodyne/*/*/'+'train'+'/*/*/*.bin')+glob.glob(root+'/data_odometry_velodyne/*/*/'+'val'+'/*/*/*.bin')

		self.label_list = [i.replace("velodyne", "labels") for i in self.lidar_list]

		self.label_list = [i.replace("bin", "label") for i in self.label_list]
	   
		print (len(self.label_list))


	def __len__(self):
		return len(self.lidar_list)

	def __getitem__(self, index):
	   
		self.A.open_scan(self.lidar_list[index])
		self.A.open_label(self.label_list[index])

		dataset_dict = {}
		#dataset_dict['xyz'] = F.to_tensor(np.expand_dims(self.A.proj_mask,axis=-1)*(self.A.proj_xyz-self.xyz_mean)/self.xyz_std)
		if self.if_perturb:
			single_random_matrix=0.97+0.06*np.random.rand(self.range_h,self.range_w).astype(np.float32)
			multi_random_matrix=np.tile(np.expand_dims(single_random_matrix,axis=-1),[1,1,3]).astype(np.float32)
			dataset_dict['xyz'] = self.A.proj_xyz*multi_random_matrix
			dataset_dict['remission'] = self.A.proj_remission*single_random_matrix
			dataset_dict['range_img'] = self.A.proj_range*single_random_matrix
		
		else:
			dataset_dict['xyz'] = self.A.proj_xyz
			dataset_dict['remission'] = self.A.proj_remission
			dataset_dict['range_img'] = self.A.proj_range
		
		if self.if_range_mask:
			range_mask=self.A.proj_range/80.0
			dataset_dict['xyz_mask'] = self.A.proj_mask*range_mask
			range_mask=None
		else:
			dataset_dict['xyz_mask'] = self.A.proj_mask
		#dataset_dict['remission'] = F.to_tensor(self.A.proj_mask*(self.A.proj_remission-self.remission_mean)/self.remission_std)
		
		semantic_label= self.A.proj_sem_label
		instance_label= self.A.proj_inst_label
		x_y_z_img=self.A.proj_xyz
		
		semantic_train_label=self.generate_label(semantic_label)

		dataset_dict['semantic_label']=semantic_train_label
		if self.if_aug:
			split_point=random.randint(100,self.range_w-100)
			dataset_dict=self.sample_transform(dataset_dict,split_point)
		rand_mask_single=None
		rand_mask_multi=None



		input_tensor,semantic_label,semantic_label_mask=self.prepare_input_label_semantic(dataset_dict)


		#sample = {'input_tensor': input_tensor, 'semantic_label': semantic_label,'semantic_label_mask':semantic_label_mask}

		return  F.to_tensor(input_tensor), F.to_tensor(semantic_label).to(dtype=torch.long), F.to_tensor(semantic_label_mask)


	def prepare_input_label_semantic(self,sample):

		if self.if_remission and not self.if_range:
			each_input=[sample['xyz'],np.expand_dims(sample['remission'],axis=-1)]
			input_tensor=np.concatenate(each_input,axis=-1)
		if self.if_remission and self.if_range:
			each_input=[sample['xyz'],np.expand_dims(sample['remission'],axis=-1),np.expand_dims(sample['range_img'],axis=-1)]
			input_tensor=np.concatenate(each_input,axis=-1)
		if not self.if_remission and not self.if_range:
			input_tensor=sample['xyz']
		semantic_label=sample['semantic_label'][:,:]
		semantic_label_mask=sample['xyz_mask'][:,:]
		return input_tensor,semantic_label,semantic_label_mask

	def sample_transform(self,dataset_dict,split_point):
	    dataset_dict['xyz']=np.concatenate([dataset_dict['xyz'][:,split_point:,:],dataset_dict['xyz'][:,:split_point,:]],axis=1)

	    dataset_dict['xyz_mask']=np.concatenate([dataset_dict['xyz_mask'][:,split_point:],dataset_dict['xyz_mask'][:,:split_point]],axis=1)

	    dataset_dict['remission']=np.concatenate([dataset_dict['remission'][:,split_point:],dataset_dict['remission'][:,:split_point]],axis=1)
	    dataset_dict['range_img']=np.concatenate([dataset_dict['range_img'][:,split_point:],dataset_dict['range_img'][:,:split_point]],axis=1)

	    dataset_dict['semantic_label']=np.concatenate([dataset_dict['semantic_label'][:,split_point:],dataset_dict['semantic_label'][:,:split_point]],axis=1)
		

	    return dataset_dict






	def sem_label_transform(self,raw_label_map):
		for i in self.label_transfer_dict.keys():
			raw_label_map[raw_label_map==i]=self.label_transfer_dict[i]
		return raw_label_map

	def generate_label(self,semantic_label):

		original_label=np.copy(semantic_label)
		label_new=self.sem_label_transform(original_label)
		
		return label_new


					

