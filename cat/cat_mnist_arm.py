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
Categorical = tf.contrib.distributions.Categorical
Dirichlet = tf.contrib.distributions.Dirichlet

#%%
def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))

def categorical_loglikelihood(b, z_concate):
    '''
    b is N*K*n_cv*n_class, one-hot vector in row
    z_concate is logits, softplus(z_concate) is prob
    z_concate is N*1*n_cv*n_class, first column 0
    '''
    lik_v = b*(z_concate-tf.reduce_logsumexp(z_concate,axis=3,keep_dims=True))
    return tf.reduce_sum(lik_v,axis=3)
        

def encoder(x,z_dim,reuse=False):
    #return logits [N,K,n_cv*(n_class-1)]
    #z_dim is n_cv*(n_class-1)
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        h2 = slim.stack(x, slim.fully_connected,[512,256],activation_fn=lrelu)
        #h1 = tf.layers.dense(2. * x - 1., 200, tf.nn.relu, name="encoder_1")
        #h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="encoder_2")
        z = tf.layers.dense(h2, z_dim, name="encoder_out",activation = None)
    return z

def decoder(b,x_dim,reuse=False):
    #return logits
    #b is [N,K,n_cv,n_class]
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        shape = b.get_shape().as_list()
        latent_dim = np.prod(shape[2:]) #equal to z_concate_dim
        b = tf.reshape(b, [-1, shape[1],latent_dim])    
        h2 = slim.stack(b, slim.fully_connected,[256,512],activation_fn=lrelu)
        #h1 = tf.layers.dense(2. * b - 1., 200, tf.nn.relu, name="decoder_1")
        #h2 = tf.layers.dense(h1, 200, tf.nn.relu, name="decoder_2")
        logit_x = tf.layers.dense(h2, x_dim, activation = None)
    return logit_x
    

def kl_cat(q_logit, p_logit):
    '''
    input: N*n_cv*n_class
    '''
    eps = 1e-10
    q = tf.nn.softmax(q_logit,dim=2)
    p = tf.nn.softmax(p_logit,dim=2)
    return tf.reduce_sum(q*(tf.log(q+eps)-tf.log(p+eps)),axis = [1,2])


def fun(x_star,E,prior_logit0,z_concate,axis_dim=2,reuse_decoder=False):
    '''
    x_star is N*K*d_x, E is N*K*n_cv*n_class, z_concate is N*n_cv*n_class
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    '''
    #KL
    #logits_py = tf.ones_like(z_concate) #uniform
    prior_logit1 = tf.expand_dims(prior_logit0,axis=0)
    logits_py = tf.tile(prior_logit1,[tf.shape(E)[0],1,1])
    
    p_cat_b = Categorical(logits=logits_py)
    q_cat_b = Categorical(logits=z_concate)
    KL_qp = tf.contrib.distributions.kl(q_cat_b, p_cat_b)
    KL = tf.reduce_sum(KL_qp,1)
    #KL_qp = kl_cat(z_concate, logits_py)
    
    #log p(x_star|E)
    logit_x = decoder(E,x_dim,reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, logit_x)
    # (N,K)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=axis_dim)
    
    return - log_p_x_given_b + KL
    

def Fn(pai,prior_logit0,z_concate,x_star_u):
    '''
    pai is [N,K_u,n_cv,n_class]
    z_concate is [N,K_u,n_class]
    '''
    z_concate1 = tf.expand_dims(z_concate,axis=1)
    E = tf.one_hot(tf.argmin(tf.log(pai+eps)-z_concate1,axis = 3),depth=n_class)
    E = tf.cast(E,tf.float32)
    return fun(x_star_u,E,prior_logit0,z_concate,reuse_decoder=True)
    
def swap(pai,j,m):
    '''
    pai is [N,K_u,n_cv,n_class]
    swap [N,K_u,n_cv,0] and [N,K_u,n_cv,m]
    '''
    depth = pai.get_shape()[-1]
    id1 = range(depth)
    id1 = tf.constant(id1)
    one_hot_m = tf.one_hot(m,depth,dtype=tf.int32)
    one_hot_j = tf.one_hot(j,depth,dtype=tf.int32)
    id2 = id1 - m*one_hot_m + j*one_hot_m
    id2 = id2 - j*one_hot_j + m*one_hot_j
    
    return tf.transpose(tf.gather(tf.transpose(pai),id2))

#def pick0(z_concate):
#    z1 = tf.expand_dims(z_concate,axis=2)
#    z2 = tf.expand_dims(z_concate,axis=3)
#    #sig = tf.sigmoid(-tf.abs(z1-z2))*(tf.exp(z1)+tf.exp(z2))/tf.reduce_sum(tf.exp(z1),axis=3,keep_dims=True)
#    
#    maxi = tf.maximum(z1,z2)
#    sig0 = 2*maxi/(tf.abs(tf.exp(z1)-tf.exp(z2))+tf.reduce_sum(tf.exp(z1),axis=3,keep_dims=True))
#    sig1 = tf.sigmoid(-tf.abs(z1-z2))
#    sig = 1 - sig0 + sig0*(1-2*sig1)
#    sig = tf.reduce_sum(sig,axis=2)
#    sig = tf.reduce_mean(sig,axis=[0,1])
#    return tf.cast(tf.argmax(sig,axis=0),tf.int32)

def pick(z_concate):
    z_ave = tf.reduce_mean(z_concate,axis=[0,1])
    return tf.cast(tf.argmax(z_ave,axis=0),tf.int32)
    
    
#%%
tf.reset_default_graph() 

x_dim = 784
n_class = 10 ; n_cv = 20  # # of classes and # of cat variables
z_dim = n_cv * (n_class-1)   # # of latent parameters neede for one cat var is n_cat-1
z_concate_dim = n_cv * n_class

eps = 1e-10
K_u = 1; K_b = 1
lr=tf.constant(0.0001)

prior_logit0 = tf.get_variable("p_b_logit", dtype=tf.float32,initializer=tf.zeros([n_cv,n_class]))

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)

N = tf.shape(x_binary)[0]

#encoder q(b|x) = log Cat(b|z_concate)
#logits for categorical, p=softmax(logits)
z0 = encoder(x_binary,z_dim)  #N*d_z
z = tf.reshape(z0,[N,n_cv,n_class-1])
zeros_logits = tf.zeros(shape = [N,n_cv,1])
z_concate = tf.concat([zeros_logits,z],axis=2) #N*n_cv*n_class
q_b = Categorical(logits=z_concate) #sample K_b \bv
#non-binary, accompanying with encoder parameter, cannot backprop
b_sample = q_b.sample(K_b) #K_b*N*n_cv
b_sample = tf.one_hot(tf.transpose(b_sample,perm=[1,0,2]),depth=n_class)  #N*K_b*n_cv
b_sample = tf.cast(b_sample,tf.float32)

#compute decoder p(x|b), gradient of encoder parameter can be automatically given by loss
x_star_b = tf.tile(tf.expand_dims(x_binary,axis=1),[1,K_b,1]) #N*K_b*d_x
#average over K_b
gen_loss0 = tf.reduce_mean(fun(x_star_b,b_sample,prior_logit0,z_concate,reuse_decoder= False),axis=1) 
gen_loss = tf.reduce_mean(gen_loss0) #average over N
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)


#provide encoder q(b|x) gradient by data augmentation
Dir = Dirichlet([1.0]*n_class)
pai = Dir.sample(sample_shape=[N,K_u,n_cv]) #[N,K_u,n_cv,n_class]

x_star_u = tf.tile(tf.expand_dims(x_binary,axis=1),[1,K_u,1]) #N*K_u*d_x

jj = 0
#jj = pick(z_concate)

pai_slice_j = tf.slice(pai,begin=[0,0,0,jj],size=[-1,-1,-1,1]) #N,K_u,n_cv,1
pai_j = swap(pai,jj,0)
F_j = Fn(pai_j,prior_logit0,z_concate,x_star_u) #N*K_u
for mm in range(1,n_class):
    pai_m = swap(pai,jj,mm)
    F_m = Fn(pai_m,prior_logit0,z_concate,x_star_u)
    grad_m = tf.expand_dims(tf.expand_dims(F_m-F_j,2),3)*(1-n_class*pai_slice_j)
    if mm == 1:
        alpha_grads = grad_m
    else:
        alpha_grads = tf.concat([alpha_grads,grad_m],3)

    
alpha_grads = tf.reduce_mean(alpha_grads,axis=1) #N*n_cv*d_b, expectation over pai
alpha_grads = tf.reshape(alpha_grads,[-1,z_dim])
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
#log_alpha_b is #N*z_dim, alpha_grads is N*z_dim, inf_vars is d_theta
#d_theta, should devide by batch-size, but can be absorb into learning rate
inf_grads = tf.gradients(z0, inf_vars, grad_ys=alpha_grads)#/b_s
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)

prior_train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(gen_loss,var_list=[prior_logit0])


with tf.control_dependencies([gen_train_op, inf_train_op, prior_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%% TRAIN
# get data
mnist = input_data.read_data_sets(os.getcwd()+'/MNIST', one_hot=True)
train_data = mnist.train
test_data = mnist.test
valid_data = mnist.validation

drectory = os.getcwd()+'/discrete_out/'
if not os.path.exists(drectory):
    os.makedirs(drectory)
  
batch_size = 100
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
        cost_eval.append(sess.run(gen_loss0,{x:xs}))
    return np.mean(cost_eval)
        
EXPERIMENT = 'MNIST_Cat_ARM'
print('Training stats....',EXPERIMENT)

sess=tf.InteractiveSession()
sess.run(init_op)
record = [];step = 0

import time
start = time.time()
COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[];time_list=[]
j_record = []
for epoch in range(training_epochs):
    avg_cost = 0.
    avg_cost_test = 0.
    np_lr = 0.0001 
    for i in range(total_batch):
        train_xs,_ = train_data.next_batch(batch_size)  
        _,cost = sess.run([train_op,gen_loss],{x:train_xs,lr:np_lr})
        record.append(cost)
        step += 1
        #j_record.append(sess.run(jj,{x:train_xs}))
        
    if epoch%1 == 0:
        valid_loss = get_loss(sess,valid_data,total_valid_batch)
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(valid_loss)
        print(epoch,'valid_cost=',valid_loss,'with std=',np.std(record))

    if epoch%5 == 0:
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list]
        cPickle.dump(all_, open(drectory+EXPERIMENT, 'w'))
    record=[]

#after training, calculate ELBO on training and testing respectively
#cost_test = get_loss(sess,test_data,total_test_batch)
#print("Final test elbo per point is", np.mean(cost_test))
#
#cost_train = get_loss(sess,train_data,total_batch)
#print("Final train elbo per point is", np.mean(cost_train))


print(EXPERIMENT)

    
    
    





