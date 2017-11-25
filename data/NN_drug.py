
# coding: utf-8
import pdb
import tensorflow as tf
import numpy as np
import pickle
import glob
import pdb
from tqdm import tqdm

#config = ConfigParser.RawConfigParser()
#config.read('config.cfg')
#which_data = config.get('Parameter', 'what_data_use')

def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

#exp_data = read_data('ccle_final_exp_data.mat')
#mut_data = read_data('ccle_final_mutation_data.mat')
sample = read_data('ccle_final_sample.name')
exp_feature = read_data('ccle_final_exp_feature')
mut_feature = read_data('ccle_features')
drug_name = read_data('ccle_drug.name')
label = read_data('ccle_drug50.label')
data = np.concatenate((mut_data, exp_data), axis=1)

def cross_validation(data, label,test_ratio,k):
	t1 = test_ratio*k
	t2 = test_ratio*(k+1)
	train_x = np.concatenate((data[:int(data.shape[0]*t1)],data[int(data.shape[0]*t2):]),axis=0)
	test_x = data[int(data.shape[0]*t1):int(data.shape[0]*t2)]
	test_sample = sample[int(data.shape[0]*t1):int(data.shape[0]*t2)]
	train_y = np.concatenate((label[:int(label.shape[0]*t1)], label[int(label.shape[0]*t2):]),axis=0)
	test_y = label[int(label.shape[0]*t1):int(label.shape[0]*t2)]
	
	print("Train: ",train_x.shape,"Test: ", test_x.shape[0])
	return train_x, test_x, train_y, test_y, test_sample

def random_batch(train_data, train_label, batch_size):
	idx = np.arange(len(train_data))
	np.random.shuffle(idx)
	x = train_data[idx]
	y = train_label[idx]
	return x[:batch_size], y[:batch_size]

print("The # of Drug Type: ", len(drug_name))


N, itr, lr, d, test_ratio = 64, 5000, 0.01, 16, 0.2
#config.getint('Parameter', 'batch_size'), config.getint('Parameter','iteration'), config.getfloat('Parameter', 'learning_rate'), config.get('Parameter', 'train_dir'), config.getint('Parameter', 'hidden_dim'), config.getfloat('Parameter', 'test_ratio'), config.getboolean('Parameter', 'l2_regularizer_use')
g_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr, g_step, 100, 0.98)
x = tf.placeholder(tf.float32, [None, data.shape[1]], name="x")
y = tf.placeholder(tf.int32, [None, len(drug_name)],name="y")

W1 = tf.Variable(tf.zeros([data.shape[1],d]), name='W1')
b1 = tf.Variable(tf.zeros(d), name="b")
y_1 = tf.matmul(x,W1) + b1

W2 = tf.Variable(tf.zeros([d,len(drug_name)]), name="W2")
b2 = tf.Variable(tf.zeros(len(drug_name)), name="b2")
y_2 = tf.matmul(y_1, W2) +b2 #logits
softmax_logits = tf.nn.softmax(y_2)
prediction = tf.argmax(softmax_logits,1) #return max idx
correct = tf.equal(prediction, tf.argmax(y,1), name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_2)
loss_mean = tf.reduce_mean(loss)
train_step = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss_mean)
saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_test_acc = []
for k in range(int(1/test_ratio)):
	sess.run(init)
	train_x, test_x, train_y, test_y, test_sample = \
				cross_validation(data, label,test_ratio,k)
	#Train
	for i in range(itr):
		batch_x, batch_y = random_batch(train_x, train_y, N)
		_, acc, loss= sess.run([train_step, accuracy, loss_mean],\
					feed_dict={x:batch_x, y:batch_y})

		if i%100 == 0:
			print("Iteration: ",i,"acc: ", acc, "loss: ", loss)

	#Test
	test_acc, test_loss, test_pred, test_label, logits = \
		sess.run([accuracy, loss_mean, prediction, y, y_2],\
					feed_dict={x:test_x, y:test_y})
	print("Fold",k,"Test accuracy: ", test_acc, ", loss: ", test_loss, "\n")
	total_test_acc.append(test_acc)

print("Average Test Accuracy: ", sum(total_test_acc)/len(total_test_acc))
