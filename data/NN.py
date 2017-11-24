
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
which_data = 'rna'
#which_data = config.get('Parameter', 'what_data_use')

data_dir = glob.glob('*' + which_data +'.mat')
#data_dir = 'tcga_'+which_data+"final_input.mat"
#label_dir = 'tcga_'+which_data+"final_label.mat"


def read_data(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data

feature = read_data('ACC_rna.feature')
cancer_type = read_data('cancer_type')

#data = read_data(data_dir)
#label = read_data(label_dir)

def cross_validation(cancer_type, data_dir,test_ratio,k):
	t1 = test_ratio*k
	t2 = test_ratio*(k+1)
	data = read_data(data_dir[0])
	label = np.zeros([data.shape[0],len(cancer_type)])
	train_x = data[:int(data.shape[0]*t1)]
	test_x = data[int(data.shape[0]*t1):]
	label[:,0] = 1
	train_y = label[:int(label.shape[0]*t1)]
	test_y = label[int(label.shape[0]*t1):]
	idx = 0
	for i,path in enumerate(data_dir[1:]):
		tmp = read_data(path)
		data = np.concatenate((data,tmp),axis=0)
		tmp_label = np.zeros([tmp.shape[0], len(cancer_type)])
		tmp_label[:,i+1] = 1
		label = np.concatenate((label,tmp_label),axis=0)
		train_x = np.concatenate((train_x,tmp[:int(tmp.shape[0]*t1)]),axis=0)
		test_x = np.concatenate((test_x,tmp[int(tmp.shape[0]*t1):]),axis=0)
		train_y = np.concatenate((train_y,tmp_label[:int(tmp_label.shape[0]*t1)]),axis=0)
		test_y = np.concatenate((test_y,tmp_label[:int(tmp_label.shape[0]*t1)]),axis=0)
	
	print("Train: ",train_y.shape,"Test: ", test_y.shape[0])
	return train_x, test_x, train_y, test_y

def random_batch(train_data, train_label, batch_size):
	idx = np.arange(len(train_x))
	np.random.shuffle(idx)
	x = train_data[idx]
	y = train_label[idx]
	return x[:batch_size], y[:batch_size]

print("The # of Cancer Type: ", len(cancer_type))


N, itr, lr, train_log, d, test_ratio = 64, 1000, 0.001, 'train_log',16, 0.2
#config.getint('Parameter', 'batch_size'), config.getint('Parameter','iteration'), config.getfloat('Parameter', 'learning_rate'), config.get('Parameter', 'train_dir'), config.getint('Parameter', 'hidden_dim'), config.getfloat('Parameter', 'test_ratio'), config.getboolean('Parameter', 'l2_regularizer_use')
g_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr, g_step, 100, 0.98)
x = tf.placeholder(tf.float32, [None, len(feature)], name="x")
y = tf.placeholder(tf.int32, [None, len(cancer_type)],name="y")

W1 = tf.Variable(tf.zeros([len(feature),d]), name='W1')
b1 = tf.Variable(tf.zeros(d), name="b")
y_1 = tf.matmul(x,W1) + b1

W2 = tf.Variable(tf.zeros([d,len(cancer_type)]), name="W2")
b2 = tf.Variable(tf.zeros(len(cancer_type)), name="b2")
y_2 = tf.matmul(y_1, W2) +b2 #logits
softmax_logits = tf.nn.softmax(y_2)
prediction = tf.argmax(softmax_logits,1) #return max idx
correct = tf.equal(prediction, tf.argmax(y,1), name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_2)
loss_mean = tf.reduce_mean(loss)
train_step = tf.train.RMSPropOptimizer(lr, 0.9).minimize(loss_mean)

tf.summary.scalar('loss', loss_mean)
tf.summary.scalar('accuracy', accuracy)
merge = tf.summary.merge_all()

saver = tf.train.Saver()
init = tf.global_variables_initializer()
sess = tf.Session()
train_writer = tf.summary.FileWriter(train_log, sess.graph)
sess.run(init)

total_test_acc = []
for k in range(int(1/test_ratio)):
	sess.run(init)
	train_x, test_x, train_y, test_y = \
				cross_validation(cancer_type, data_dir,test_ratio,k)

	#Train
	for i in range(itr):
		batch_x, batch_y = random_batch(train_x, train_y, N)
		_, acc, loss, train_sum = \
				sess.run([train_step, accuracy, loss_mean, merge])

		if i%100 == 0:
			print("Iteration: ",i,"acc: ", acc, "loss: ", loss)
			train_writer.acc_summary(train_sum, i)

	#Test
	test_acc, test_loss, test_pred, test_label, logits = \
		sess.run([accuracy, loss_mean, prediction, y, y_2],\
					feed_dict={x:test_x, y:test_y})
	print("Fold",k,"Test accuracy: ", test_acc, ", loss: ", test_loss, "\n")
	total_test_acc.append(test_acc)

print("Average Test Accuracy: ", sum(total_test_acc)/len(total_test_acc))
