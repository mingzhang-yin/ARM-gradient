
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import os

#from OneHotCategorical import *
#from RelaxedOneHotCategorical import *

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
OneHotCategorical = tf.contrib.distributions.OneHotCategorical
RelaxedOneHotCategorical = tf.contrib.distributions.RelaxedOneHotCategorical
Categorical = tf.contrib.distributions.Categorical
# In[ ]:


batch_size = 100
tau0=1.0 # initial temperature

K=10 # number of classes
N=200//K # number of categorical distributions
straight_through=True # if True, use Straight-through Gumbel-Softmax
kl_type='relaxed' # choose between ('relaxed', 'categorical')
learn_temp=False 


# In[ ]:


x0 = tf.placeholder(tf.float32, shape=(batch_size,784), name='x')
x = tf.to_float(x0 > .5)
#net = tf.cast(tf.random_uniform(tf.shape(x)) < x, x.dtype) # dynamic binarization
net = slim.stack(x,slim.fully_connected,[512,256])
logits_y = tf.reshape(slim.fully_connected(net,K*N,activation_fn=None),[-1,N,K])
tau = tf.Variable(tau0,name="temperature",trainable=learn_temp)
q_y = RelaxedOneHotCategorical(tau,logits_y)
y = q_y.sample()
if straight_through:
  y_hard = tf.cast(tf.one_hot(tf.argmax(y,-1),K), y.dtype)
  y = tf.stop_gradient(y_hard - y) + y
net1 = slim.flatten(y)
net1 = slim.stack(net1,slim.fully_connected,[256,512])
logits_x = slim.fully_connected(net1,784,activation_fn=None)
p_x = Bernoulli(logits=logits_x)


recons = tf.reduce_sum(p_x.log_prob(x),1)
logits_py = tf.ones_like(logits_y) * 1./K #uniform

if kl_type=='categorical' or straight_through:
  # Analytical KL with Categorical prior
  p_cat_y = Categorical(logits=logits_py)
  q_cat_y = Categorical(logits=logits_y)
  KL_qp = tf.contrib.distributions.kl(q_cat_y, p_cat_y)
else:
  # Monte Carlo KL with Relaxed prior
  p_y = RelaxedOneHotCategorical(tau,logits=logits_py)
  KL_qp = q_y.log_prob(y) - p_y.log_prob(y)


lr=tf.constant(0.0001)

KL = tf.reduce_sum(KL_qp,1)
mean_recons = tf.reduce_mean(recons)
mean_KL = tf.reduce_mean(KL)
loss = -tf.reduce_mean(recons - KL)


train_op=tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init_op=tf.global_variables_initializer()


#%%
mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data = mnist.train
test_data = mnist.test
valid_data = mnist.validation

directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
  
total_points = mnist.train.num_examples
total_batch = int(total_points / batch_size)
total_test_batch = int(mnist.test.num_examples / batch_size)
total_valid_batch = int(mnist.validation.num_examples / batch_size)

training_epochs = 1000
display_step = total_batch

#%%
def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(KL-recons,{x:xs}))
    return np.mean(cost_eval)
        
EXPERIMENT = 'MNIST_Cat_VAE_gumbel'
print('Training stats....',EXPERIMENT)

sess=tf.InteractiveSession()
sess.run(init_op)
record = [];step = 0

import time
start = time.time()
COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[];time_list=[]

for epoch in range(training_epochs):
    np_lr = 0.0001
    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size)  
        _,cost,_ = sess.run([train_op,loss,tau],{x0:train_xs,lr:np_lr})
        record.append(cost)
        step += 1
        
        
    if epoch%1 == 0:
        valid_loss = get_loss(sess,valid_data,total_valid_batch)
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(valid_loss)
        print(epoch,'valid_cost=',valid_loss,'with std=',np.std(record))
    if epoch%5 == 0:
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
    record=[]

#after training, calculate ELBO on training and testing respectively
cost_test = get_loss(sess,test_data,total_test_batch)
print("Final test elbo per point is", np.mean(cost_test))

cost_train = get_loss(sess,train_data,total_batch)
print("Final train elbo per point is", np.mean(cost_train))


print(EXPERIMENT)


import cPickle
all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list]
cPickle.dump(all_, open(directory+EXPERIMENT, 'w'))















