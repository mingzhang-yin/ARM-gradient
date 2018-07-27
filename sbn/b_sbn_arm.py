#This code can only train with M==1, because when M=1, lower bound and the true object are same
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
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def encoder1(x,bi_dim,reuse=False):
    #return logits #Eric Jang uses [512,256]
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        h1 = slim.stack(x, slim.fully_connected,[300],activation_fn=lrelu,scope='encoder_0')      
        log_alpha1 = tf.layers.dense(h1, bi_dim, name="encoder_1")
    return log_alpha1

def encoder2(b1,bi_dim,reuse=False):
    #return logits #Eric Jang uses [512,256]
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        h2 = slim.stack(b1, slim.fully_connected,[300],activation_fn=lrelu,scope='encoder_2')
        log_alpha2 = tf.layers.dense(h2, bi_dim, name="encoder_3")
    return log_alpha2

def decoder(b2,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        h3 = slim.stack(b2, slim.fully_connected,[300],activation_fn=lrelu)
        log_alphax = tf.layers.dense(h3, x_dim, None, name="decoder_1")
    return log_alphax




def fun(x_lower,E2,axis_dim=2,reuse_encoder=False,reuse_decoder=False):
    '''
    x_star,E are N*(d_x or 2*d_bi)
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    return LogLik,  N*M
    '''    
    #log p(x_star|E), x_dim is global
    log_alpha_x = decoder(E2,x_dim,reuse=reuse_decoder)
    log_p_x_given_b2 = bernoulli_loglikelihood(x_lower, log_alpha_x)
    log_p_x_given_b2 = tf.reduce_sum(log_p_x_given_b2, axis=axis_dim)
    
    
    return -log_p_x_given_b2
    
def fig_gnrt(figs,epoch,show=False,bny=True):
    '''
    input:N*28*28
    '''
    sns.set_style("whitegrid", {'axes.grid' : False})
    plt.ioff()
    
    nx = ny = 10
    canvas = np.empty((28*ny, 28*nx))
    for i in range(nx):
        for j in range(ny):
            canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = figs[i*nx+j]
    
    plt.figure(figsize=(8, 10))        
    plt.imshow(canvas, origin="upper", cmap="gray")
    plt.tight_layout()
    path = os.getcwd()+'/out/'
    if not os.path.exists(path):
        os.makedirs(path)
    name_fig = path + str(epoch)+'.png'
    plt.savefig(name_fig, bbox_inches='tight') 
    plt.close('all')
    if show:
        plt.show()   
    

#%%
tf.reset_default_graph() 

bi_dim = 200
b_dim = 2*bi_dim; x_dim = 392
eps = 1e-10
# number of sample b to calculate gen_loss, 
# number of sample u to calculate inf_grads

lr=tf.constant(0.0001)
M = tf.placeholder(tf.int32) 


x_u = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
xu_binary = tf.to_float(x_u > .5)
xu_star = tf.tile(tf.expand_dims(xu_binary,axis=1),[1,M,1]) #N*M*d_x


x_l = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
xl_binary = tf.to_float(x_l > .5)
xl_star = tf.tile(tf.expand_dims(xl_binary,axis=1),[1,M,1]) #N*M*d_x

N = tf.shape(xu_binary)[0]

#encoder q(b|x) = log Ber(b|log_alpha_b)
#logits for bernoulli, p=sigmoid(logits)
log_alpha_b1 = encoder1(xu_star,bi_dim)  #N*d_b 
q_b1 = Bernoulli(logits=log_alpha_b1) #sample 1 \bv
b_sample1 = q_b1.sample() #M*N*d_b, accompanying with encoder parameter, cannot backprop
b_sample1 = tf.cast(b_sample1,tf.float32) #N*M*d_b

log_alpha_b2 = encoder2(b_sample1 ,bi_dim) 
q_b2 = Bernoulli(logits=log_alpha_b2) #sample 1 \bv
b_sample2 = q_b2.sample() #N*d_b, accompanying with encoder parameter, cannot backprop
b_sample2 = tf.cast(b_sample2,tf.float32) #N*M*d_b

#compute decoder p(x|b), gradient of decoder parameter can be automatically given by loss
gen_loss00 = fun(xl_star,b_sample2,reuse_encoder=True,reuse_decoder= False)
gen_loss0 = tf.reduce_mean(gen_loss00,1) #average over M
gen_loss = tf.reduce_mean(gen_loss0) #average over N
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)

b2_onesample = tf.reshape(tf.slice(b_sample2,[0,0,0],[-1,1,-1]),[-1,bi_dim])
xl_recon_alpha = decoder(b2_onesample,x_dim,reuse=True)
xl_dist = Bernoulli(logits = xl_recon_alpha)
xl_recon = xl_dist.mean()
x_recon = tf.concat([x_u,xl_recon],axis=1)
x_recon = tf.reshape(x_recon,[-1,28,28])

#for M>1, true loss
loss0 = -(tf.reduce_logsumexp(-gen_loss00,axis=1) -tf.log(tf.cast(M,tf.float32)))
loss = tf.reduce_mean(loss0)


#provide encoder q(b|x) gradient by data augmentation

#gradient to log_alpha_b2(phi_2) 
u_noise2 = tf.random_uniform(shape=[N,M,bi_dim],maxval=1.0)
P2_1 = tf.sigmoid(-log_alpha_b2)
E2_1 = tf.cast(u_noise2>P2_1,tf.float32)
P2_2 = 1 - P2_1
E2_2 = tf.cast(u_noise2<P2_2,tf.float32)

F2_1 = fun(xl_star,E2_1,reuse_encoder=True,reuse_decoder=True) #N*M
F2_2 = fun(xl_star,E2_2,reuse_encoder=True,reuse_decoder=True)
alpha2_grads = tf.expand_dims(F2_1-F2_2,axis=2)*(u_noise2-0.5) #N*M*d_bi
#alpha2_grads = tf.reduce_mean(alpha2_grads,axis=1)


#gradient to log_alpha_b1(phi_1) 
u_noise1 = tf.random_uniform(shape=[N,M,bi_dim],maxval=1.0)
P1_1 = tf.sigmoid(-log_alpha_b1)
E1_1 = tf.cast(u_noise1>P1_1,tf.float32)
P1_2 = 1 - P1_1
E1_2 = tf.cast(u_noise1<P1_2,tf.float32)

log_alpha_b2_1 = encoder2(E1_1 ,bi_dim,reuse=True) 
q_b2_1 = Bernoulli(log_alpha_b2_1) #sample 1 \bv
b_sample2_1 = q_b2_1.sample() #N*M*d_bi, accompanying with encoder parameter, cannot backprop
b_sample2_1 = tf.cast(b_sample2_1,tf.float32) #N*M*d_bi
F1_1 = fun(xl_star,b_sample2_1,reuse_encoder=True,reuse_decoder=True) #N*M

log_alpha_b2_2 = encoder2(E1_2 ,bi_dim,reuse=True) 
q_b2_2 = Bernoulli(log_alpha_b2_2) #sample 1 \bv
b_sample2_2 = q_b2_2.sample() #N*M*d_bi, accompanying with encoder parameter, cannot backprop
b_sample2_2 = tf.cast(b_sample2_2,tf.float32) #N*M*d_bi
F1_2 = fun(xl_star,b_sample2_2,reuse_encoder=True,reuse_decoder=True) #N*M

alpha1_grads = tf.expand_dims(F1_1-F1_2,axis=2)*(u_noise1-0.5) #N*M*d_bi




inf_opt = tf.train.AdamOptimizer(lr)
log_alpha_b = tf.concat([log_alpha_b1,log_alpha_b2],2)
alpha_grads = tf.concat([alpha1_grads,alpha2_grads],2)
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
inf_grads = tf.gradients(log_alpha_b, inf_vars, grad_ys=alpha_grads)
inf_gradvars = zip(inf_grads, inf_vars)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)



with tf.control_dependencies([gen_train_op,inf_train_op]):
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
  
batch_size = 100
total_points = mnist.train.num_examples
total_batch = int(total_points / batch_size)
total_test_batch = int(mnist.test.num_examples / batch_size)
total_valid_batch = int(mnist.validation.num_examples / batch_size)

training_epochs = 2000
display_step = total_batch
#%%
def get_loss(sess,data,total_batch,M0):
    cost_eval = []                  
    for j in range(total_batch):
        xs,_ = data.next_batch(batch_size)
        xs_u = xs[:,:392]; xs_l = xs[:,392:]
        cost_eval.append(sess.run(loss0,{x_u:xs_u,x_l:xs_l,M:M0}))
    return np.mean(cost_eval)


EXPERIMENT = 'SBN_MNIST_Bernoulli_ARM'
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
        x_upper = train_xs[:,:392]; x_lower = train_xs[:,392:]
        #plt.imshow(np.reshape(x_upper[0,:],[14,28]))
        _,cost = sess.run([train_op,gen_loss],{x_u:x_upper,x_l:x_lower,lr:np_lr,M:1})
        record.append(cost)
        step += 1
    if epoch%1 == 0:
        valid_loss = get_loss(sess,valid_data,total_valid_batch,M0=1)
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(valid_loss)
        print(epoch,'valid_loss=',valid_loss)
    if epoch%5 == 0:
        test_loss = get_loss(sess,test_data,total_test_batch,M0=1000)
        COST_TEST.append(test_loss)  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
        print(epoch,'test_loss=',test_loss)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'w'))
        
        x_re = sess.run(x_recon,{x_u:x_upper,M:1000})
        fig_gnrt(x_re,epoch,bny=0)
    record=[]
            
##after training, calculate ELBO on training and testing respectively
#cost_test = get_loss(sess,test_data,total_test_batch)
#print("Final test elbo per point is", np.mean(cost_test))
#
#cost_train = get_loss(sess,train_data,total_batch)
#print("Final train elbo per point is", np.mean(cost_train))

print(EXPERIMENT)












