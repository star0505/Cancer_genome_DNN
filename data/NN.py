
# coding: utf-8
import pdb
import tensorflow as tf
import numpy as np
import pickle
import glob
import config
import ConfigParser
import pdb
from tqdm import tqdm

config = ConfigParser.RawConfigParser()
config.read('config.cfg')

which_data = config.get('Parameter', 'what_data_use')

data_dir = glob.glob('*' + what_data +'.mat')

def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

feature = read_data(data_dir[0])
cancer_type = read_data('cancer_type')

def cross_validation(cancer_type, data_dir,test_ratio,k):
	t1 = test_ratio*k
	t2 = test_ratio*(k+1)
	label = np.zeros([data.shape[0],len(cancer_type)])
	data = read_data(data_dir[0])
	train_x = data[:int(len(data))*t1]
	test_x
	label[:,0] = 1
	idx = 0
	for i,path in enumerate(data_dir[1:]):
		tmp = read_data(path)
		data = np.concatenate((data,tmp),axis=0)
		tmp_label = np.zeros([len(tmp), len(cancer_type)])
		tmp_label[:,i+1] = 1
		label = np.concatenate((label,tmp_label),axis=0)
		train_x = np.concatenate((),axis=0)

print("Load Data: ", data.shape)
print("The # of Cancer Type: ", len(cancer_type))


N, itr, lr, train_log, d, test_ratio = config.getint('Parameter', 'batch_size'), config.getint('Parameter','iteration'), config.getfloat('Parameter', 'learning_rate'), config.get('Parameter', 'train_dir'), config.getint('Parameter', 'hidden_dim'), config.getfloat('Parameter', 'test_ratio'), config.getboolean('Parameter', 'l2_regularizer_use')
g_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr, g_step, 100, 0.98)
x = tf.placeholder(tf.float32, [None, len(feature)], name="x")
y = tf.placeholder(tf.int32, [None, len(cancer_type)],name="y")

W1 = tf.Variable(tf.zeros([len(feature),d]), name='W1')
b1 = tf.Variable(tf.zeros(b), name="b")
y_1 = tf.matmul(x,W1) + b1

W2 = tf.Variable(tf.zeros([d,len(cancer_type)]), name="W2")
b2 = tf.Variable(tf.zeros(len(cancer_type)), name="b2")
y_2 = tf.matmul(y_1, W2) +b2
softmax_logits = tf.nn.softmax(y_2)
prediction = tf.argmax(softmax_logits,1)
correct = tf.equal(prediction, tf.argmax(y,1), name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_2)
loss_mean = tf.reduce_mean(loss)
train_step = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss_mean)

tf.summary.scalar('loss', loss_mean)

