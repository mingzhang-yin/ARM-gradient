from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import matplotlib
matplotlib.use('Agg')

import numpy as np
import os
import sys
import seaborn as sns
import scipy.spatial.distance
from matplotlib import pyplot as plt
import pandas as pd 
import scipy.stats as stats
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cPickle

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli

#%%
def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))

def lrelu(x, alpha=0.2):
    #weak nonlinearity, larger alpha, weaker nonlinearity
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def encoder(x,b_dim,reuse=False):
    #return logits #Eric Jang uses [512,256]
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        h2 = slim.stack(x, slim.fully_connected,[200,200],activation_fn=lrelu)
#        h1 = tf.layers.dense(2. * x - 1., 200, tf.nn.relu, name="encoder_1")
#        h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="encoder_2")
        log_alpha = tf.layers.dense(h2, b_dim, activation=None)
    return log_alpha


def decoder(b,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        h2 = slim.stack(b ,slim.fully_connected,[200,200],activation_fn=lrelu)
        #h1 = tf.layers.dense(2. * b - 1., 200, tf.nn.relu, name="decoder_1")
        #h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="decoder_2")
        log_alpha = tf.layers.dense(h2, x_dim, activation=None)
    return log_alpha


def fun(x_star,E,axis_dim=2,reuse_encoder=False,reuse_decoder=False):
    '''
    x_star,E are N*K*(d_x or d_b)
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    return elbo,  N*K
    '''
    #log q(E|x_star), b_dim is global
    log_alpha_b = encoder(x_star,b_dim,reuse=reuse_encoder)
    log_q_b_given_x = bernoulli_loglikelihood(E, log_alpha_b)
    # (N,K),conditional independent d_b Bernoulli
    log_q_b_given_x = tf.reduce_sum(log_q_b_given_x , axis=axis_dim)

    #log p(E)
    log_p_b = bernoulli_loglikelihood(E, tf.zeros_like(E)) 
    log_p_b = tf.reduce_sum(log_p_b, axis=axis_dim)
    
    #log p(x_star|E), x_dim is global
    log_alpha_x = decoder(E,x_dim,reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, log_alpha_x)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=axis_dim)
    
    return log_q_b_given_x - (log_p_x_given_b + log_p_b) 
    
    
    

#%%
tf.reset_default_graph() 

b_dim = 200; x_dim = 784
eps = 1e-10
# number of sample b to calculate gen_loss, 
# number of sample u to calculate inf_grads
K_u = 1; K_b = 1
lr=tf.constant(0.0001)

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

#encoder q(b|x) = log Ber(b|log_alpha_b)
#logits for bernoulli, p=sigmoid(logits)
log_alpha_b = encoder(x_binary,b_dim)  #N*d_b 

q_b = Bernoulli(logits=log_alpha_b) #sample K_b \bv
b_sample = q_b.sample(K_b) #K_b*N*d_b, accompanying with encoder parameter, cannot backprop
b_sample = tf.cast(tf.transpose(b_sample,perm=[1,0,2]),tf.float32) #N*K_b*d_b

#compute decoder p(x|b), gradient of decoder parameter can be automatically given by loss
x_star_b = tf.tile(tf.expand_dims(x_binary,axis=1),[1,K_b,1]) #N*K_b*d_x
#average over K_b
gen_loss0 = tf.reduce_mean(fun(x_star_b,b_sample,reuse_encoder=True,reuse_decoder= False),axis=1) 
gen_loss = tf.reduce_mean(gen_loss0) #average over N
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)

#provide encoder q(b|x) gradient by data augmentation
u_noise = tf.random_uniform(shape=[N,K_u,b_dim],maxval=1.0) #sample K \uv
P1 = tf.tile(tf.expand_dims(tf.sigmoid(-log_alpha_b),axis=1),[1,K_u,1])
E1 = tf.cast(u_noise>P1,tf.float32)
P2 = 1 - P1
E2 = tf.cast(u_noise<P2,tf.float32)
x_star_u = tf.tile(tf.expand_dims(x_binary,axis=1),[1,K_u,1]) #N*K_u*d_x

#N*K_u
F1 = fun(x_star_u,E1,reuse_encoder=True,reuse_decoder=True)
F2 = fun(x_star_u,E2,reuse_encoder=True,reuse_decoder=True)

alpha_grads = tf.expand_dims(F1-F2,axis=2)*(u_noise-0.5) #N*K_u*d_b
alpha_grads = tf.reduce_mean(alpha_grads,axis=1) #N*d_b, expectation over u
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
#log_alpha_b is N*d_b, alpha_grads is N*d_b, inf_vars is d_theta
#d_theta; should be devided by batch-size, but can be absorbed into learning rate
inf_grads = tf.gradients(log_alpha_b, inf_vars, grad_ys=alpha_grads)#/b_s
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)


with tf.control_dependencies([gen_train_op, inf_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%% TRAIN
# get data
mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data = mnist.train
test_data = mnist.test
valid_data = mnist.validation

directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
  
batch_size = 25
total_points = mnist.train.num_examples
total_batch = int(total_points / batch_size)
total_test_batch = int(mnist.test.num_examples / batch_size)
total_valid_batch = int(mnist.validation.num_examples / batch_size)

training_epochs = 400
display_step = total_batch
#%%
def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)  
        cost_eval.append(sess.run(gen_loss0,{x:xs}))
    return np.mean(cost_eval)


EXPERIMENT = 'MNIST_Bernoulli_ARM'
print('Training stats....',EXPERIMENT)

sess=tf.InteractiveSession()
sess.run(init_op)
record = []; step = 0

import time
start = time.time()
COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[];time_list=[]

for epoch in range(training_epochs):
    avg_cost = 0.
    avg_cost_test = 0.
    np_lr = 0.0001

    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size)  
        _,cost = sess.run([train_op,gen_loss],{x:train_xs,lr:np_lr})
        record.append(cost)
        step += 1
    print(epoch,'cost=',np.mean(record),'with std=',np.std(record))
    if epoch%1 == 0:
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(get_loss(sess,valid_data,total_valid_batch))
    if epoch%5 == 0:
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'w'))
    record=[]
            
#after training, calculate ELBO on training and testing respectively
#cost_test = get_loss(sess,test_data,total_test_batch)
#print("Final test elbo per point is", np.mean(cost_test))
#
#cost_train = get_loss(sess,train_data,total_batch)
#print("Final train elbo per point is", np.mean(cost_train))

print(EXPERIMENT)












