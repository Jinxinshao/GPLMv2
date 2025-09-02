import os
import random
import numpy as np
import cv2

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., only_h_flip=False):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	# simple re-weight for the edge
	if random.random() < Hc / H * edge_decay:
		Hs = 0 if random.randint(0, 1) == 0 else H - Hc
	else:
		Hs = random.randint(0, max(0, H - Hc))
		#print(H-Hc)
		#Hs = random.randint(0, H-Hc)

	if random.random() < Wc / W * edge_decay:
		Ws = 0 if random.randint(0, 1) == 0 else W - Wc
	else:
		Ws = random.randint(0,max(0, W-Wc))
		#Ws = random.randint(0, W-Wc)

	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	# horizontal flip
	if random.randint(0, 1) == 1:
		for i in range(len(imgs)):
			imgs[i] = np.flip(imgs[i], axis=1)

	if not only_h_flip:
		# bad data augmentations for outdoor
		rot_deg = random.randint(0, 3)
		for i in range(len(imgs)):
			imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))
			
	return imgs


def align(imgs=[], size=256):
	H, W, _ = imgs[0].shape
	Hc, Wc = [size, size]

	Hs = (H - Hc) // 2
	Ws = (W - Wc) // 2
	for i in range(len(imgs)):
		imgs[i] = imgs[i][Hs:(Hs+Hc), Ws:(Ws+Wc), :]

	return imgs


class PairLoader(Dataset):
	def __init__(self, data_dir, sub_dir, mode, size=256, edge_decay=0, only_h_flip=False):
		assert mode in ['train', 'valid', 'test']

		self.mode = mode
		self.size = size
		self.edge_decay = edge_decay
		self.only_h_flip = only_h_flip

		self.root_dir = os.path.join(data_dir, sub_dir)
		self.img_names = self.get_image_file_names(os.path.join(self.root_dir, 'GT'))
		#self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'GT')))
		self.img_num = len(self.img_names)
		print(self.img_num)

	def get_image_file_names(self, directory):
		all_file_names = sorted(os.listdir(directory))
		valid_image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
		img_names = [file_name for file_name in all_file_names if os.path.splitext(file_name)[1].lower() in valid_image_extensions]
		return img_names

    # Filter out only the file names with valid image extensions
    
    

    

	def __len__(self):
		return self.img_num

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]
		#print(img_name)
		#source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
		#print(source_img)
		#target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
		#print(source_img)
		valid_image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
		if os.path.splitext(img_name)[1].lower() in valid_image_extensions:
			source_img = read_img(os.path.join(self.root_dir, 'hazy', img_name)) * 2 - 1
			target_img = read_img(os.path.join(self.root_dir, 'GT', img_name)) * 2 - 1
			TM_img = read_img(os.path.join(self.root_dir, 'TM', img_name)) * 2 - 1
			TM_all_img = read_img(os.path.join(self.root_dir, 'TM_all', img_name)) * 2 - 1
			TM_allT_img = read_img(os.path.join(self.root_dir, 'TM_allT', img_name)) * 2 - 1
			TM_D_img = read_img(os.path.join(self.root_dir, 'TM_D', img_name)) * 2 - 1
			TM_SAM = read_img(os.path.join(self.root_dir, 'SAM', img_name)) * 2 - 1

			if self.mode == 'train':
				[source_img, target_img, TM_img, TM_all_img, TM_allT_img, TM_D_img, TM_SAM] = augment([source_img, target_img,TM_img, TM_all_img, TM_allT_img, TM_D_img, TM_SAM], self.size, self.edge_decay, self.only_h_flip)

			if self.mode == 'valid':
				[source_img, target_img, TM_img, TM_all_img, TM_allT_img, TM_D_img, TM_SAM] = align([source_img, target_img,TM_img, TM_all_img, TM_allT_img, TM_D_img, TM_SAM], self.size)

		return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img),'TM': hwc_to_chw(TM_img),'TM_all': hwc_to_chw(TM_all_img) ,'TM_allT': hwc_to_chw(TM_allT_img) ,'TM_D': hwc_to_chw(TM_D_img),'TM_SAM': hwc_to_chw(TM_SAM), 'filename': img_name}


class SingleLoader(Dataset):
	def __init__(self, root_dir):
		self.root_dir = root_dir
		self.img_names = self.get_image_file_names(os.listdir(self.root_dir))
		self.img_num = len(self.img_names)

	def __len__(self):
		return self.img_num

	def get_image_file_names(self, directory):
		all_file_names = sorted(os.listdir(directory))
		valid_image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.gif']
		img_names = [file_name for file_name in all_file_names if os.path.splitext(file_name)[1].lower() in valid_image_extensions]
		return img_names

	def modify_path(root_dir, img_name):
		dir_name, file_name = os.path.split(img_name)
		dirs = dir_name.split(os.sep)
		dirs[-1] = 'TM'
		new_dir_name = os.path.join(root_dir, os.sep.join(dirs), file_name)
		return new_dir_name

	def modify_path_TMall(root_dir, img_name):
		dir_name, file_name = os.path.split(img_name)
		dirs = dir_name.split(os.sep)
		dirs[-1] = 'TM_all'
		new_dir_name = os.path.join(root_dir, os.sep.join(dirs), file_name)
		return new_dir_name


	
	def modify_path_TMallT(root_dir, img_name):
		dir_name, file_name = os.path.split(img_name)
		dirs = dir_name.split(os.sep)
		dirs[-1] = 'TM_allT'
		new_dir_name = os.path.join(root_dir, os.sep.join(dirs), file_name)
		return new_dir_name

	def modify_path_D(root_dir, img_name):
		dir_name, file_name = os.path.split(img_name)
		dirs = dir_name.split(os.sep)
		dirs[-1] = 'TM_D'
		new_dir_name = os.path.join(root_dir, os.sep.join(dirs), file_name)
		return new_dir_name	


	def modify_path_SAM(root_dir, img_name):
		dir_name, file_name = os.path.split(img_name)
		dirs = dir_name.split(os.sep)
		dirs[-1] = 'SAM'
		new_dir_name = os.path.join(root_dir, os.sep.join(dirs), file_name)
		return new_dir_name	

	def __getitem__(self, idx):
		cv2.setNumThreads(0)
		cv2.ocl.setUseOpenCL(False)

		# read image, and scale [0, 1] to [-1, 1]
		img_name = self.img_names[idx]

		img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1
		TM = read_img((self.modify_path(self.root_dir), img_name))* 2 - 1
		TM_all = read_img((self.modify_path_TMall(self.root_dir), img_name))* 2 - 1
		TM_allT = read_img((self.modify_path_TMallT(self.root_dir), img_name))* 2 - 1
		TM_D = read_img((self.modify_path_D(self.root_dir), img_name))* 2 - 1
		TM_SAM = read_img((self.modify_path_SAM(self.root_dir), img_name))* 2 - 1
		return {'img': hwc_to_chw(img),'TM':hwc_to_chw(TM), 'TM_all':hwc_to_chw(TM_all), 'TM_allT':hwc_to_chw(TM_allT), 'TM_D':hwc_to_chw(TM_D), 'TM_SAM':hwc_to_chw(TM_SAM),'filename': img_name}
