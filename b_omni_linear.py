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
import scipy.io
from tensorflow.examples.tutorials.mnist import input_data
import cPickle

slim=tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli

#%%
def bernoulli_loglikelihood(b, log_alpha):
    return b * (-tf.nn.softplus(-log_alpha)) + (1 - b) * (-log_alpha - tf.nn.softplus(-log_alpha))

def lrelu(x, alpha=0.2):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)



def encoder(x,b_dim,reuse=False):
    #return logits #Eric Jang uses [512,256]
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        log_alpha = tf.layers.dense( x, b_dim, None, name="encoder_1")
    return log_alpha

def decoder(b,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        log_alpha = tf.layers.dense(b, x_dim, None, name="decoder_1")
    return log_alpha


def fun(x_star,E,prior_logit0,axis_dim=2,reuse_encoder=False,reuse_decoder=False):
    '''
    x_star,E are N*K*(d_x or d_b)
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    x_star is observe x; E is latent b
    return elbo,  N*K
    '''
    #log q(E|x_star), b_dim is global
    log_alpha_b = encoder(x_star,b_dim,reuse=reuse_encoder)
    log_q_b_given_x = bernoulli_loglikelihood(E, log_alpha_b)
    # (N,K),conditional independent d_b Bernoulli
    log_q_b_given_x = tf.reduce_sum(log_q_b_given_x , axis=axis_dim)

    #log p(E)
    prior_logit1 = tf.expand_dims(tf.expand_dims(prior_logit0,axis=0),axis=0)
    prior_logit = tf.tile(prior_logit1,[tf.shape(E)[0],tf.shape(E)[1],1])
    log_p_b = bernoulli_loglikelihood(E, prior_logit) 
    log_p_b = tf.reduce_sum(log_p_b, axis=axis_dim)
    
    #log p(x_star|E), x_dim is global
    log_alpha_x = decoder(E,x_dim,reuse=reuse_decoder)
    log_p_x_given_b = bernoulli_loglikelihood(x_star, log_alpha_x)
    log_p_x_given_b = tf.reduce_sum(log_p_x_given_b, axis=axis_dim)
    #-ELBO
    return log_q_b_given_x - (log_p_x_given_b + log_p_b) 
    

def load_omniglot(data_file=os.getcwd()+'/omniglot.mat'):
      """Reads in Omniglot images.
    
      Args:
        binarize: whether to use the fixed binarization
    
      Returns:
        x_train: training images
        x_valid: validation images
        x_test: test images
    
      """
      n_validation=1345
    
      def reshape_data(data):
        return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
      omni_raw = scipy.io.loadmat(data_file)
    
      train_data = reshape_data(omni_raw['data'].T.astype('float32'))
      test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
    
      # Binarize the data with a fixed seed
      np.random.seed(5)
      train_data = (np.random.rand(*train_data.shape) < train_data).astype(float)
      test_data = (np.random.rand(*test_data.shape) < test_data).astype(float)
    
      shuffle_seed = 123
      permutation = np.random.RandomState(seed=shuffle_seed).permutation(train_data.shape[0])
      train_data = train_data[permutation]
    
      x_train = train_data[:-n_validation]
      x_valid = train_data[-n_validation:]
      x_test = test_data
    
      return x_train, x_valid, x_test   
  
def evidence(sess,data,elbo, batch_size = 100, S = 100, total_batch = None):
    '''
    For correct use:
    ELBO for x_i must be calculated by SINGLE z sample from q(z|x_i)
    '''
    #from scipy.special import logsumexp    
    if total_batch is None:
        total_batch = int(data.num_examples / batch_size)
        
    avg_evi = 0
    for j in range(total_batch):
        test_xs = data.next_batch(batch_size)         
        elbo_accu = np.empty([batch_size,0])
        for i in range(S):
            elbo_i = sess.run(elbo,{x:test_xs})
            elbo_accu = np.append(elbo_accu,elbo_i,axis=1)
        
        evi0 = sess.run(tf.reduce_logsumexp(elbo_accu,axis = 1))
        evi = np.mean(evi0 - np.log(S))
        avg_evi += evi / total_batch
    return avg_evi 
    

#%%
tf.reset_default_graph() 

b_dim = 200; x_dim = 784
eps = 1e-10
# number of sample b to calculate gen_loss, 
# number of sample u to calculate inf_grads
K_u = 1; K_b = 1
lr=tf.constant(0.001)

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)
prior_logit0 = tf.get_variable("p_b_logit", dtype=tf.float32,initializer=tf.zeros([b_dim]))

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
neg_elbo = tf.reduce_mean(fun(x_star_b,b_sample,prior_logit0,reuse_encoder=True,reuse_decoder= False),axis=1) [:,np.newaxis]
gen_loss = tf.reduce_mean(neg_elbo) #average over N
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
F1 = fun(x_star_u,E1,prior_logit0,reuse_encoder=True,reuse_decoder=True)
F2 = fun(x_star_u,E2,prior_logit0,reuse_encoder=True,reuse_decoder=True)

alpha_grads = tf.expand_dims(F1-F2,axis=2)*(u_noise-0.5) #N*K_u*d_b
alpha_grads = tf.reduce_mean(alpha_grads,axis=1) #N*d_b, expectation over u
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
#log_alpha_b is N*d_b, alpha_grads is N*d_b, inf_vars is d_theta
#d_theta; should be devided by batch-size, but can be absorbed into learning rate
inf_grads = tf.gradients(log_alpha_b, inf_vars, grad_ys=alpha_grads)#/b_s
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr,beta1=0.5,beta2=.99999)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)

prior_train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(gen_loss,var_list=[prior_logit0])



with tf.control_dependencies([gen_train_op, inf_train_op,prior_train_op]):
#with tf.control_dependencies([gen_train_op, inf_train_op]):
    train_op = tf.no_op()
    
init_op=tf.global_variables_initializer()

#%% TRAIN
# get data
train,valid,test = load_omniglot()

from preprocess import preprocess
train_data = preprocess(train)

test_data = preprocess(test[:8000])

valid_data = preprocess(valid[:1300])


directory = os.getcwd()+'/discrete_out/'
if not os.path.exists(directory):
    os.makedirs(directory)
  
batch_size = 25
total_points = train.shape[0]
total_batch = int(total_points / batch_size)
total_test_batch = int(test.shape[0] / batch_size)
total_valid_batch = int(valid.shape[0] / batch_size)

training_epochs = 1200
display_step = total_batch


#%%
def get_loss(sess,data,total_batch):
    cost_eval = []                  
    for j in range(total_batch):
        xs = data.next_batch(batch_size)  
        cost_eval.append(sess.run(neg_elbo,{x:xs}))
    return np.mean(cost_eval)


np_lr = 0.0001
np.random.seed()
EXPERIMENT = 'OMNI_Bernoulli_ARM' + '_linear_'+str(int(np.random.randint(0,100,1)))
print('Training starts....',EXPERIMENT)
print('Learning rate....',np_lr)

sess=tf.InteractiveSession()
sess.run(init_op)
step = 0

import time
start = time.time()
COUNT=[]; COST=[]; TIME=[];COST_TEST=[];COST_VALID=[];epoch_list=[];time_list=[]
evidence_r = [];
all_ = []
for epoch in range(training_epochs):
    record=[]

    for i in range(total_batch):
        train_xs = train_data.next_batch(batch_size)  
        _,cost = sess.run([train_op,gen_loss],{x:train_xs,lr:np_lr})
        record.append(cost)
        step += 1
    print(epoch,'cost=',np.mean(record),'with std=',np.std(record))
    if epoch%1 == 0:
        COUNT.append(step); COST.append(np.mean(record)); TIME.append(time.time()-start)
        COST_VALID.append(get_loss(sess,valid_data,total_valid_batch))
        
    if epoch%5 == 0:
        avg_evi_val = evidence(sess, valid_data, -neg_elbo, batch_size, S = 100, total_batch=10)
        print(epoch,'The validation NLL is', -np.round(avg_evi_val,2))
        evidence_r.append(np.round(avg_evi_val,2))
        
        COST_TEST.append(get_loss(sess,test_data,total_test_batch))  
        epoch_list.append(epoch)
        time_list.append(time.time()-start)
        all_ = [COUNT,COST,TIME,COST_TEST,COST_VALID,epoch_list,time_list,evidence_r]
        cPickle.dump(all_, open(directory+EXPERIMENT, 'wb'))

print(EXPERIMENT)









