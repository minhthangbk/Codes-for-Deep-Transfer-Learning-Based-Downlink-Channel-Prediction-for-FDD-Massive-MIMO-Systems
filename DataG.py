import numpy as np
import math
import cmath

import scipy.io as scio
# data preprocessing
# ant=64
# diff=2
# dataNew = './DataSave/samples_source_noised' + str(ant) + '_1593_' + str(diff) + '.mat'
# data_matrix = scio.loadmat(dataNew)
# Source_Task_list = data_matrix['Source_Task_list']
# dataNew = './DataSave/samples_target_noised' + str(ant) + '_800_' + str(diff) + '.mat'
# # dataNew = './DataSave/samples_targetwhy2'+str(ant)+'_592_'+str(diff)+'.mat'
# data_matrix = scio.loadmat(dataNew)
# Target_Task_list = data_matrix['Target_Task_list']
#
# print(Source_Task_list.shape)
# print(Target_Task_list.shape)
# Source_Task_list = Source_Task_list[0:1500,0 :40, :, :]
# mean_channel = np.zeros((1, 1, 2, np.shape(Source_Task_list)[-1]), dtype=float)
# max_channel = np.zeros((1, 1, 2, np.shape(Source_Task_list)[-1]), dtype=float)
# min_channel = np.zeros((1, 1, 2, np.shape(Source_Task_list)[-1]), dtype=float)
# for channel_pairs in [Source_Task_list, Target_Task_list]:
# 	mean_channel += np.mean(np.mean(channel_pairs, axis=0, keepdims=True), axis=1, keepdims=True) / 2
# 	max_channel += np.max(np.max(channel_pairs, axis=0, keepdims=True), axis=1, keepdims=True) / 2
# 	min_channel += np.min(np.min(channel_pairs, axis=0, keepdims=True), axis=1, keepdims=True) / 2
#
# Source_Task_list_norm = (Source_Task_list - mean_channel) / (max_channel - min_channel)
# Target_Task_list_norm = (Target_Task_list - mean_channel) / (max_channel - min_channel)
#
# dataNew = './DataSave/samples_source64_2.mat'
# scio.savemat(dataNew,{'Source_Task_list_norm':Source_Task_list_norm})
# dataNew = './DataSave/samples_target64_2.mat'
# scio.savemat(dataNew,{'Target_Task_list_norm':Target_Task_list_norm})


dataNew = './DataSave/samples_target64_2.mat'
data=scio.loadmat(dataNew)
Target_Task_list_norm=data['Target_Task_list_norm']

class UpDownTaskGenerator(object):

	def __init__(self,datasize,diff,ant):
		self.supportrate = 0.5
		# self.ImportNormlizeData(datasize,diff,ant)
		self.num_target_tasks=np.shape(Target_Task_list_norm )[0]

	def source_data(self,num_train):

		# batch_task_index=np.random.randint(0, high=self.num_source_tasks, size=(1,))[0]
		# batch_task=self.Source_Task_list_norm[batch_task_index]
		# data_index_set = np.arange(len(batch_task))
		# np.random.shuffle(data_index_set)
		times=1
		dataNew = './DataSave/samples_source64_2.mat'
		data = scio.loadmat(dataNew)
		Source_Task_list_norm = data['Source_Task_list_norm']
		self.num_source_tasks=np.shape(Source_Task_list_norm)[0]
		batch_task_index=np.random.randint(0, high=self.num_source_tasks, size=(20,))

		batch_task=Source_Task_list_norm[batch_task_index]
		bat,samp,dim2,num_ant=batch_task.shape
		batch_task=batch_task.reshape([bat*samp,dim2,num_ant])

		data_index_set = np.arange(len(batch_task))
		np.random.shuffle(data_index_set)

		batch_support_set=batch_task[data_index_set[0:int(self.supportrate * len(batch_task) )]]
		batch_query_set=batch_task[data_index_set[int(self.supportrate * len(batch_task) ):]]
		batch_support_channels=batch_support_set[np.random.randint(0,high=len(batch_support_set), size=num_train*times)]
		batch_query_channels=batch_query_set[np.random.randint(0,high=len(batch_query_set), size=num_train*times)]

		# print('batch_support_channels.shape: ',batch_support_channels.shape)
		h_up_support_batch=[]
		h_down_support_batch=[]
		h_up_query_batch=[]
		h_down_query_batch=[]
		for batch_index in range(num_train*times):
			h_up_support_batch.append(batch_support_channels[batch_index,0,:])
			h_down_support_batch.append(batch_support_channels[batch_index,1,:])
			h_up_query_batch.append(batch_query_channels[batch_index,0,:])
			h_down_query_batch.append(batch_query_channels[batch_index,1,:])
		h_up_support_batch=np.asarray(h_up_support_batch)
		h_down_support_batch=np.asarray(h_down_support_batch)
		h_up_query_batch=np.asarray(h_up_query_batch)
		h_down_query_batch=np.asarray(h_down_query_batch)
		# print('h_up_support_batch.shape: ',h_up_support_batch.shape)
		return h_up_support_batch,h_down_support_batch,h_up_query_batch,h_down_query_batch

	def target_data(self,num_eval,index):
		# batch_task_index=np.random.randint(0, high=self.num_target_tasks, size=(1,))[0]
		batch_task=Target_Task_list_norm[index]
		data_index_set = np.arange(len(batch_task))
		# np.random.shuffle(data_index_set)

		batch_support_set=batch_task[data_index_set[0:int(self.supportrate * len(batch_task) )]]
		batch_query_set=batch_task[data_index_set[int(self.supportrate * len(batch_task) ):]]
		batch_support_channels=batch_support_set[np.random.randint(0,high=len(batch_support_set), size=num_eval)]
		batch_query_channels=batch_query_set[np.random.randint(0,high=len(batch_query_set), size=num_eval)]
		# print('batch_support_channels.shape: ',batch_support_channels.shape)
		h_up_support_batch=[]
		h_down_support_batch=[]
		h_up_query_batch=[]
		h_down_query_batch=[]
		for batch_index in range(num_eval):
			h_up_support_batch.append(batch_support_channels[batch_index,0,:])
			h_down_support_batch.append(batch_support_channels[batch_index,1,:])
			h_up_query_batch.append(batch_query_channels[batch_index,0,:])
			h_down_query_batch.append(batch_query_channels[batch_index,1,:])
		h_up_support_batch=np.asarray(h_up_support_batch)
		h_down_support_batch=np.asarray(h_down_support_batch)
		h_up_query_batch=np.asarray(h_up_query_batch)
		h_down_query_batch=np.asarray(h_down_query_batch)
		return h_up_support_batch,h_down_support_batch,h_up_query_batch,h_down_query_batch


	def org_dis(self, x_support, y_support):
		return np.mean(np.power(x_support - y_support, 2)) / np.mean(np.power(y_support, 2))
