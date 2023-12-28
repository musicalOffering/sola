import random
import pickle
import numpy as np
import torch.utils.data as data
import os
import os.path as osp

def get_first_index(length, size):
	assert length - size >= 0, f'length - size = {length-size}'
	max_first = length - size
	first = random.randint(0, max_first)
	return first

class VideoPretrain(data.Dataset):
	def __init__(self, subset='training', config=None):
		if subset not in ['training', 'validation', 'testing']:
			raise Exception('invalid subset')
		self.subset = subset
		self.config = config
		data_root = config['data_root']
		with open(osp.join(data_root ,f'{subset}_duration.pkl'), 'rb') as f:
			duration_dict = pickle.load(f)
		data_directory = osp.join(data_root, subset)
		self.file_list = []
		for file_name in os.listdir(data_directory):
			full_path = osp.join(data_directory, file_name)
			if duration_dict[file_name] > config['sampling_duration']:
				self.file_list.append(full_path)
		print("="*40)
		print("Dataset Init Complete.")
		print("Total number of video : ", len(self.file_list))
		print("Minimum duration of feature: ", config['sampling_duration'])
		print("Step: ", config['step'])
		print("TSM size: ", config['tsm_size'])
		print("="*40,'\n')

	def __len__(self):
		return len(self.file_list)

	def __getitem__(self, index):
		file_name = self.file_list[index]
		video = np.load(file_name, mmap_mode='r')
		length = len(video)
		size = self.config['sampling_duration']
		first = get_first_index(length, size)
		video = video[first:first+size]
		assert len(video) == size, f'len(video):{len(video)}'
		return video

class VideoEvaluation(data.Dataset):
	def __init__(self, subset, config=None, load_model=''):
		super().__init__()
		if subset not in ['training', 'validation', 'testing']:
			raise Exception('invalid subset')
		self.subset = subset
		self.config = config
		if load_model == 'process_subset':
			self.process_subset = True
		else:
			print(f'load_model: {load_model}')
			self.process_subset = False
		if self.process_subset:
			data_root = osp.join(config['feature_save_path'], f'{config["exp_name"]}_val')
		else:
			data_root = osp.join(config['feature_save_path'], f'{config["exp_name"]}_{load_model}')
		self.positive_root = osp.join(data_root, f'{subset}_1')
		self.negative_root = osp.join(data_root, f'{subset}_0')
		self.positive_len = len(os.listdir(self.positive_root))
		self.data_len = self.positive_len + len(os.listdir(self.negative_root))
		#print('VidEvaluation dataset setup complete')

	def __len__(self):
		return self.data_len

	def __getitem__(self, index):
		sample_index = random.randint(0,99)
		if index >= self.positive_len:
			index -= self.positive_len
			data = np.load(osp.join(self.negative_root, f'{index}.npy'))[sample_index]
			label = 0
		else:
			data = np.load(osp.join(self.positive_root, f'{index}.npy'))[sample_index]
			label = 1
		return data, label