import tensorflow as tf
import numpy as np


class MAMLModel(object):

	def __init__(self, name, sess, number_of_antennas, grad_steps, eval_grad_steps, inner_lr, meta_lr, fo=False):
		self.name = name
		with tf.variable_scope(self.name):
			self.input_dim = number_of_antennas * 2
			self.output_dim = self.input_dim
			self.hidden_1 = 128#number_of_antennas * 2
			self.hidden_2 = 128#number_of_antennas * 2
			self.inner_lr = inner_lr
			self.meta_lr = meta_lr
			self.is_fo = fo
			self.grad_steps = grad_steps
			self.eval_grad_steps = eval_grad_steps  # 判断是否在grad_steps时候停下优化效果最好
			self.eval_grad_steps_fix=0
			self.weights = self.build_model()
			self.train_build_ops()
			self.sess = sess
			self.test_sgd_ops()
			self.test_adam_ops()

	def build_model(self):

		self.inputs_support = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)
		self.labels_support = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float32)
		self.inputs_query = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)
		self.labels_query = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float32)

		self.eval_inputs_support = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)
		self.eval_labels_support = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float32)
		self.eval_inputs_query = tf.placeholder(shape=[None, self.input_dim], dtype=tf.float32)
		self.eval_labels_query = tf.placeholder(shape=[None, self.output_dim], dtype=tf.float32)
		self.adam=tf.placeholder(dtype=tf.float32)


		self.ep = tf.Variable(
			0,
			dtype=tf.int32,
			name='episodes',
			trainable=False
		)
		self.inc_ep = self.ep.assign_add(1)

		weights = {}

		# hidden_1
		weights['w_hidden_1'] = tf.Variable(
			tf.truncated_normal([self.input_dim, self.hidden_1], stddev=tf.sqrt(2 / self.input_dim)), dtype=tf.float32)
		weights['b_hidden_1'] = tf.Variable(tf.zeros([self.hidden_1]), dtype=tf.float32)

		# hidden_2
		weights['w_hidden_2'] = tf.Variable(
			tf.truncated_normal([self.hidden_1, self.hidden_2], stddev=tf.sqrt(2 / self.hidden_1), dtype=tf.float32))
		weights['b_hidden_2'] = tf.Variable(tf.zeros([self.hidden_2]), dtype=tf.float32)

		# output
		weights['w_output'] = tf.Variable(
			tf.truncated_normal([self.hidden_2, self.output_dim], stddev=tf.sqrt(2 / self.hidden_2)), dtype=tf.float32)
		weights['b_output'] = tf.Variable(tf.truncated_normal([self.output_dim], stddev=0.1), dtype=tf.float32)

		return weights

	def forwardprop(self, x, weights):
		hidden = x
		for i in range(2):
			hidden = tf.nn.relu(
				tf.matmul(hidden, weights['w_hidden_{}'.format(i + 1)]) + weights['b_hidden_{}'.format(i + 1)])
		output = tf.matmul(hidden, weights['w_output']) + weights['b_output']
		return output

	def train_build_ops(self):

		self.train_support_losses = []
		self.train_query_losses = []
		loss_support = tf.losses.mean_squared_error(self.labels_support, self.forwardprop(self.inputs_support,
																						  self.weights)) / tf.reduce_mean(
			tf.square(
				self.labels_support))  # tf.losses.mean_squared_error(self.labels_support , np.zeros(self.batch_size,self.output_dim))
		loss_query = tf.losses.mean_squared_error(self.labels_query,
												  self.forwardprop(self.inputs_query, self.weights)) / tf.reduce_mean(
			tf.square((self.labels_query)))
		self.train_support_losses.append(loss_support)
		self.train_query_losses.append(loss_query)
		grads = tf.gradients(loss_support, list(self.weights.values()))
		grads, _ = tf.clip_by_global_norm(grads, 40.0)
		grads = dict(zip(self.weights.keys(), grads))
		fast_weights = dict(zip(self.weights.keys(), [weight_item - self.inner_lr * grads[key] for key, weight_item in
													  self.weights.items()]))

		for i in np.arange(self.grad_steps - 1):
			loss_support = tf.losses.mean_squared_error(self.labels_support, self.forwardprop(self.inputs_support,
																							  fast_weights)) / tf.reduce_mean(
				tf.square(self.labels_support))
			loss_query = tf.losses.mean_squared_error(self.labels_query, self.forwardprop(self.inputs_query,
																						  fast_weights)) / tf.reduce_mean(
				tf.square(self.labels_query))
			self.train_support_losses.append(loss_support)
			self.train_query_losses.append(loss_query)
			grads = tf.gradients(loss_support, list(fast_weights.values()))
			grads, _ = tf.clip_by_global_norm(grads, 40.0)
			grads = dict(zip(fast_weights.keys(), grads))
			fast_weights = dict(zip(fast_weights.keys(),
									[weight_item - self.inner_lr * grads[key] for key, weight_item in
									 fast_weights.items()]))

		loss_support = tf.losses.mean_squared_error(self.labels_support, self.forwardprop(self.inputs_support,
																						  fast_weights)) / tf.reduce_mean(
			tf.square(self.labels_support))
		loss_query = tf.losses.mean_squared_error(self.labels_query,
												  self.forwardprop(self.inputs_query, fast_weights)) / tf.reduce_mean(
			tf.square(self.labels_query))
		self.train_support_losses.append(loss_support)
		self.train_query_losses.append(loss_query)

		self.loss_query_inner = loss_query
		self.loss_support_inner = loss_support
		optimizer = tf.train.AdamOptimizer(self.meta_lr)
		self.metatrain_op = optimizer.minimize(loss_query)

	def test_sgd_ops(self):
		self.loss_support_eval = tf.losses.mean_squared_error(self.eval_labels_support , self.forwardprop(self.eval_inputs_support, self.weights) )/tf.reduce_mean(tf.square(self.eval_labels_support))
		self.loss_query_eval= tf.losses.mean_squared_error(self.eval_labels_query , self.forwardprop(self.eval_inputs_query, self.weights) )/tf.reduce_mean(tf.square(self.eval_labels_query))

		grads = tf.gradients(self.loss_support_eval , list(self.weights.values()))
		grads = dict(zip(self.weights.keys(), grads))
		self.copy = [tf.assign(self.weights[key], weight_item - self.inner_lr* grads[key]) for key, weight_item in self.weights.items()]


	def test_adam_ops(self):

		self.loss_support_eval_adam = tf.losses.mean_squared_error(self.eval_labels_support , self.forwardprop(self.eval_inputs_support, self.weights) )/tf.reduce_mean(tf.square(self.eval_labels_support))
		self.loss_query_eval_adam= tf.losses.mean_squared_error(self.eval_labels_query , self.forwardprop(self.eval_inputs_query, self.weights) )/tf.reduce_mean(tf.square(self.eval_labels_query))
		optimizer_test = tf.train.AdamOptimizer(self.adam)
		self.metatest_op = optimizer_test.minimize(self.loss_support_eval_adam)

	def train(self, x_support, y_support, x_query, y_query):
		print("Meta-training MAML {} Task #{}...".format(self.name, self.sess.run(self.ep)))

		loss_support, loss_query, _ = self.sess.run([self.loss_support_inner, self.loss_query_inner, self.metatrain_op],
													feed_dict={self.inputs_support: x_support,
															   self.labels_support: y_support,
															   self.inputs_query: x_query,
															   self.labels_query: y_query,
															   })
		print(loss_support, ' ', loss_query)
		self.sess.run(self.inc_ep)

		return loss_support, loss_query


	def test(self, x_support_eval, y_support_eval, x_query_eval, y_query_eval):

		test_support_losses = []
		test_query_losses = []
		# query_losses, support_losses = self.sess.run([self.loss_query_eval, self.loss_support_eval],
		# 											 feed_dict={self.eval_inputs_support: x_support_eval,
		# 														self.eval_labels_support: y_support_eval,
		# 														self.eval_inputs_query: x_query_eval,
		# 														self.eval_labels_query: y_query_eval})
		# test_support_losses.append(support_losses)
		# test_query_losses.append(query_losses)

		for i in np.arange(self.eval_grad_steps_fix):
			_,query_losses, support_losses = self.sess.run([self.copy,self.loss_query_eval, self.loss_support_eval],
														 feed_dict={self.eval_inputs_support: x_support_eval,
																	self.eval_labels_support: y_support_eval,
																	self.eval_inputs_query: x_query_eval,
																	self.eval_labels_query: y_query_eval})
			test_support_losses.append(support_losses)
			test_query_losses.append(query_losses)
		for i in np.arange(self.eval_grad_steps-self.eval_grad_steps_fix+1):

			inner_lr = 7e-6
			# if i<=120:
			# 	inner_lr= 1e-6
			# elif i<200:
			# 	inner_lr = (-i)*5e-9+1e-6
			# else:
			# 	inner_lr = 1e-20
			# elif i<=300:
			# 	inner_lr=1e-6
			# else:
			# 	inner_lr = 1e-7
			_, query_losses, support_losses = self.sess.run([self.metatest_op, self.loss_query_eval, self.loss_support_eval],
				feed_dict={self.eval_inputs_support: x_support_eval,
						   self.eval_labels_support: y_support_eval,
						   self.eval_inputs_query: x_query_eval,
						   self.eval_labels_query: y_query_eval,
						   self.adam:inner_lr})
			test_support_losses.append(support_losses)
			test_query_losses.append(query_losses)

		return test_query_losses, test_support_losses  # /np.mean(np.power(np.linalg.norm(y_query_eval,axis=1),2))


	def org_dis(self, x_support, y_support):
		return np.mean(np.power(x_support - y_support, 2)) / np.mean(np.power(y_support, 2))