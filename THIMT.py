import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime
import time
seed=3
np.random.seed(seed)
# from UpDownTaskGenerator import UpDownTaskGenerator
# from Baseline import BaselineModel
from MetaModel import MAMLModel
from DataG import UpDownTaskGenerator
import os
import scipy.io as scio


info='MTTHthi2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

if __name__ == "__main__":

	Number_antenna = 64
	k = 20
	grad_steps = 3
	eval_grad_steps = grad_steps *400
	inner_lr = 1e-3
	meta_lr = 1e-3
	episodes = 10000
	diff=2

	task_dist = UpDownTaskGenerator(2*k,diff,Number_antenna) #support and query : 2*k

	with tf.Session() as sess:
		models = {
			"meta": MAMLModel("meta", sess, Number_antenna, grad_steps, eval_grad_steps , inner_lr, meta_lr)
		}
		sess.run(tf.global_variables_initializer())
		saver = tf.train.Saver()
		losses_support_train={}
		losses_query_train={}
		# for i in np.arange(episodes):
		# 	x_support, y_support ,x_query, y_query = task_dist.source_data(k)
		# 	for model_name, model in models.items():
		# 		if model_name not in losses_support_train:
		# 			losses_support_train[model_name] = []
		# 			losses_query_train[model_name] = []
		# 		loss_support_temp, loss_query_temp=model.train(x_support,y_support,x_query, y_query)
		#
		# 		losses_support_train[model_name].append(loss_support_temp)
		# 		losses_query_train[model_name].append(loss_query_temp)
		# saver.save(sess, save_path='./ModelSave/'+info+str(episodes)+str(diff) + str(inner_lr) + str(meta_lr) + str(
		# 	grad_steps) + str(k) + str(Number_antenna)  + '.ckpt')
		n_tests = 100
		k=20
		# 1,2,3,4,5,  8,10,  15,20,30,  40,  50,60,  70,80
		losses_query = {}
		losses_support = {}
		# outputs = {}
		for i in np.arange(n_tests):
			print("Testing with Task #{}".format(i + 1))
			saver.restore(sess, save_path='./ModelSave/'+info+str(episodes)+str(diff)  + str(inner_lr) + str(meta_lr) + str(
			grad_steps) + str(20) + str(Number_antenna)  + '.ckpt')
			asa='./ModelSave/'+info+str(episodes)+str(diff)  + str(inner_lr) + str(meta_lr) + str(grad_steps) + str(20) + str(Number_antenna)  + '.ckpt'
			print('Restore successfully',asa)
			x_support_eval, y_support_eval ,x_query_eval, y_query_eval  = task_dist.target_data(k,i)
			for model_name, model in models.items():
				if model_name not in losses_query:
					losses_query[model_name] = []
					losses_support[model_name] = []
				# outputs[model_name]=[]
				model_eval_query_losses, model_eval_support_losses = model.test(x_support_eval, y_support_eval,
																				x_query_eval, y_query_eval)
				losses_query[model_name].append(model_eval_query_losses)
				losses_support[model_name].append(model_eval_support_losses)
				# print(np.mean(y_query_eval),' ',np.mean(y_support_eval))
				print(' 0 model_eval_query_losses, model_eval_support_losses: ',model_eval_query_losses[0], model_eval_support_losses[0])
				print(' -1 model_eval_query_losses, model_eval_support_losses: ',model_eval_query_losses[-1], model_eval_support_losses[-1])

		dataNew = './ResultSave/'+info+str(k)+str(diff) +str(inner_lr)+str(meta_lr)+str(episodes)+str(grad_steps)+str(eval_grad_steps)+str(k)+str(Number_antenna)+'.mat'
		scio.savemat(dataNew, {'losses_support_train': losses_support_train, 'losses_query_train': losses_query_train,'losses_query':losses_query,'losses_support':losses_support})
		print('Data saved in'+dataNew)
		'''testing pic'''
		mean_losses_query = {}
		for model_name, mean_loss in losses_query.items():
			if model_name not in mean_losses_query:
				mean_losses_query[model_name] = np.mean(mean_loss, axis=0)


		mean_losses_support = {}
		for model_name, mean_loss in losses_support.items():
			if model_name not in mean_losses_support:
				mean_losses_support[model_name] = np.mean(mean_loss, axis=0)


		fig, ax = plt.subplots()
		for model_name, mean_loss in mean_losses_query.items():
			print(mean_loss[0],mean_loss[-1])
			ax.semilogy(np.arange(eval_grad_steps + 1), mean_loss.flatten(), label=model_name + 'query')

		for model_name, mean_loss in mean_losses_support.items():
			ax.semilogy(np.arange(eval_grad_steps + 1), mean_loss.flatten(), label=model_name + 'support')

		ax.legend()

		ax.set_title(info+str(diff) + str(inner_lr) + str(meta_lr) + str(episodes) + str(grad_steps) + str(
			eval_grad_steps) + str(k) + str(Number_antenna), fontsize=12)
		plt.show()

