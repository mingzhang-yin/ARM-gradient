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



def encoder1(x,bi_dim,reuse=False):
    #return logits #Eric Jang uses [512,256]
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        #h2 = slim.stack(x, slim.fully_connected,[400,400],activation_fn=lrelu)        
        log_alpha1 = tf.layers.dense(x, bi_dim, name="encoder_1")
    return log_alpha1

def encoder2(b1,bi_dim,reuse=False):
    #return logits #Eric Jang uses [512,256]
    with tf.variable_scope("encoder") as scope:
        if reuse:
            scope.reuse_variables()
        log_alpha2 = tf.layers.dense(b1, bi_dim, name="encoder_2")
    return log_alpha2


def decoder1(b1,x_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        log_alphax = tf.layers.dense(b1, x_dim, None, name="decoder_1")
    return log_alphax

def decoder2(b2,bi_dim,reuse=False):
    #return logits
    with tf.variable_scope("decoder") as scope:
        if reuse:
            scope.reuse_variables()
        log_alphab1 = tf.layers.dense(b2, bi_dim, None, name="decoder_2")
    return log_alphab1




def fun(x_star,E1,E2,prior_logit0,axis_dim=1,reuse_encoder=False,reuse_decoder=False):
    '''
    x_star,E are N*(d_x or 2*d_bi)
    calculate log p(x_star|E) + log p(E) - log q(E|x_star)
    axis_dim is axis for d_x or d_b
    x_star is observe x; E is latent b
    return elbo,  N
    '''
    
    #log q(E|x_star), b_dim is global
    log_alpha_b1 = encoder1(x_star,bi_dim,reuse=reuse_encoder)
    log_q_b1_given_x = bernoulli_loglikelihood(E1, log_alpha_b1)
    # (N,K),conditional independent d_b Bernoulli
    log_q_b1_given_x = tf.reduce_sum(log_q_b1_given_x , axis=axis_dim)
    
    log_alpha_b2 = encoder2(E1 ,bi_dim,reuse=reuse_encoder) 
    log_q_b2_given_b1 = bernoulli_loglikelihood(E2, log_alpha_b2)
    # (N,K),conditional independent d_b Bernoulli
    log_q_b2_given_b1 = tf.reduce_sum(log_q_b2_given_b1, axis=axis_dim)

    #log p(E)
    prior_logit1 = tf.expand_dims(prior_logit0,axis=0)
    prior_logit = tf.tile(prior_logit1,[tf.shape(E2)[0],1])
    log_p_b = bernoulli_loglikelihood(E2, prior_logit) 
    log_p_b = tf.reduce_sum(log_p_b, axis=axis_dim)
    
    
    #log p(x_star|E), x_dim is global
    log_alpha_x = decoder1(E1,x_dim,reuse=reuse_decoder)
    log_p_x_given_b1 = bernoulli_loglikelihood(x_star, log_alpha_x)
    log_p_x_given_b1 = tf.reduce_sum(log_p_x_given_b1, axis=axis_dim)
    
    log_alpha_b1 = decoder2(E2,bi_dim,reuse=reuse_decoder)
    log_p_b1_given_b2 = bernoulli_loglikelihood(E1, log_alpha_b1)
    log_p_b1_given_b2 = tf.reduce_sum(log_p_b1_given_b2, axis=axis_dim)
    
    return log_q_b1_given_x+log_q_b2_given_b1 - log_p_x_given_b1 - log_p_b1_given_b2 - log_p_b 
    

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

bi_dim = 200
b_dim = 2*bi_dim; x_dim = 784
eps = 1e-10
# number of sample b to calculate gen_loss, 
# number of sample u to calculate inf_grads

lr=tf.constant(0.0001)

x = tf.placeholder(tf.float32,[None,x_dim]) #N*d_x
x_binary = tf.to_float(x > .5)

prior_logit0 = tf.get_variable("p_b_logit", dtype=tf.float32,initializer=tf.zeros([bi_dim]))


N = tf.shape(x_binary)[0]

#encoder q(b|x) = log Ber(b|log_alpha_b)
#logits for bernoulli, p=sigmoid(logits)
log_alpha_b1 = encoder1(x_binary,bi_dim)  #N*d_b 
q_b1 = Bernoulli(logits=log_alpha_b1) #sample 1 \bv
b_sample1 = q_b1.sample() #N*d_b, accompanying with encoder parameter, cannot backprop
b_sample1 = tf.cast(b_sample1,tf.float32) #N*d_b

log_alpha_b2 = encoder2(b_sample1 ,bi_dim) 
q_b2 = Bernoulli(logits=log_alpha_b2) #sample 1 \bv
b_sample2 = q_b2.sample() #N*d_b, accompanying with encoder parameter, cannot backprop
b_sample2 = tf.cast(b_sample2,tf.float32) #N*d_b

#compute decoder p(x|b), gradient of decoder parameter can be automatically given by loss

neg_elbo = fun(x_binary,b_sample1,b_sample2,prior_logit0,reuse_encoder=True,reuse_decoder= False)[:,np.newaxis]
gen_loss = tf.reduce_mean(neg_elbo) #average over N
gen_opt = tf.train.AdamOptimizer(lr)
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
gen_gradvars = gen_opt.compute_gradients(gen_loss, var_list=gen_vars)
gen_train_op = gen_opt.apply_gradients(gen_gradvars)



#provide encoder q(b|x) gradient by data augmentation
#u_noise = tf.random_uniform(shape=[N,b_dim],maxval=1.0) #sample K \uv
#u_noise1 = tf.slice(u_noise,begin=[0,0],size=[-1,bi_dim])
#u_noise2 = tf.slice(u_noise,begin=[0,bi_dim],size=[-1,bi_dim])

#gradient to log_alpha_b2(phi_2) 
u_noise2 = tf.random_uniform(shape=[N,bi_dim],maxval=1.0)
P2_1 = tf.sigmoid(-log_alpha_b2)
E2_1 = tf.cast(u_noise2>P2_1,tf.float32)
P2_2 = 1 - P2_1
E2_2 = tf.cast(u_noise2<P2_2,tf.float32)

F2_1 = fun(x_binary,b_sample1,E2_1,prior_logit0,reuse_encoder=True,reuse_decoder=True) #N,
F2_2 = fun(x_binary,b_sample1,E2_2,prior_logit0,reuse_encoder=True,reuse_decoder=True)
alpha2_grads = tf.expand_dims(F2_1-F2_2,axis=1)*(u_noise2-0.5) #N*d_b

#gradient to log_alpha_b1(phi_1) 
u_noise1 = tf.random_uniform(shape=[N,bi_dim],maxval=1.0)
P1_1 = tf.sigmoid(-log_alpha_b1)
E1_1 = tf.cast(u_noise1>P1_1,tf.float32)
P1_2 = 1 - P1_1
E1_2 = tf.cast(u_noise1<P1_2,tf.float32)

log_alpha_b2_1 = encoder2(E1_1 ,bi_dim,reuse=True) 
q_b2_1 = Bernoulli(log_alpha_b2_1) #sample 1 \bv
b_sample2_1 = q_b2_1.sample() #N*d_b, accompanying with encoder parameter, cannot backprop
b_sample2_1 = tf.cast(b_sample2_1,tf.float32) #N*d_bi
F1_1 = fun(x_binary,E1_1,b_sample2_1,prior_logit0,reuse_encoder=True,reuse_decoder=True) #N,

log_alpha_b2_2 = encoder2(E1_2 ,bi_dim,reuse=True) 
q_b2_2 = Bernoulli(log_alpha_b2_2) #sample 1 \bv
b_sample2_2 = q_b2_2.sample() #N*d_b, accompanying with encoder parameter, cannot backprop
b_sample2_2 = tf.cast(b_sample2_2,tf.float32) #N*d_b
F1_2 = fun(x_binary,E1_2,b_sample2_2,prior_logit0,reuse_encoder=True,reuse_decoder=True) #N,

alpha1_grads = tf.expand_dims(F1_1-F1_2,axis=1)*(u_noise1-0.5) #N*d_bi

alpha_grads = tf.concat([alpha1_grads,alpha2_grads],axis=1)
inf_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='encoder')
log_alpha_b = tf.concat([log_alpha_b1,log_alpha_b2],axis=1)
#log_alpha_b is N*d_b, alpha_grads is N*d_b, inf_vars is d_theta
#d_theta; should be devided by batch-size, but can be absorbed into learning rate
inf_grads = tf.gradients(log_alpha_b, inf_vars, grad_ys=alpha_grads)#/b_s
inf_gradvars = zip(inf_grads, inf_vars)
inf_opt = tf.train.AdamOptimizer(lr)
inf_train_op = inf_opt.apply_gradients(inf_gradvars)


prior_train_op = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(gen_loss,var_list=[prior_logit0])


with tf.control_dependencies([gen_train_op,inf_train_op,prior_train_op]):
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
EXPERIMENT = 'OMNI_Bernoulli_ARM' + '_2layer_'+str(int(np.random.randint(0,100,1)))
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












