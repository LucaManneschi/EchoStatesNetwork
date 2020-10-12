
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
from Echo_activity import *
from Adam_manual import *

# MNIST Dataset, input and desired output for the classification

from tensorflow.examples.tutorials.mnist import input_data
mnist=input_data.read_data_sets("MNIST_data/", one_hot=True)
X_tr=mnist.train.images
Y_tr=mnist.train.labels
X_te=mnist.test.images
Y_te=mnist.test.labels
X_val=mnist.validation.images
Y_val=mnist.validation.labels

# Permutation of the digits for the psMNIST

P=np.shape(X_tr)[1]
permutation=np.random.permutation(P)

# Data for the classification
X_tr=X_tr[:,permutation]
X_te=X_te[:,permutation]
X_val=X_val[:,permutation]

N_out=10                # Dimensionality of the output

# ESN parameters and definition
# Select Connected=True for the hierarchical structure, and Connected=False for the parallel ESNs
# ESN number 1 is the 'deeper' reservoir in the hierarchical structure (in the parallel structure there is no difference between the 2 ESN)
# ESN number 2 is the reservoir that receives the input in the hierarchical structure

N=1200                  # Total number of nodes

N1=np.int(N/2)          # Nodes of the first reservoir
N2=np.int(N/2)          # Nodes of the second reservoir


pho1=0.985              # Value of rho for ESN 1
pho2=1                  # Value of rho for ESN 2

N1_av=5                 # Average number of connections for one node in the reservoir 1
N2_av=3                 # Average number of connections for one node in the reservoir 2

diluition1=1-N1_av/N1   # Probability of a zero in the connectivity matrix of ESN 1
diluition2=1-N2_av/N2   # Probability of a zero in the connectivity matrix of ESN 2


Connected=True
if Connected==False:

                                                    # Unconnected reservoirs (Parallel ESNs)
    W1_in=np.ones([N1])
    W1_in[np.random.uniform(0,1,N1)<0.5]=-1         # Input Connectivity matrix for ESN 1, that for a 1-d input we set usually as a random 1,-1 mask
    W1_in=gamma_In*W1_in

    W2_in=np.ones([N2])                             # Input Connectivity matrix for ESN 2, that for a 1-d input we set usually as a random 1,-1 mask
    W2_in[np.random.uniform(0,1,N2)<0.5]=-1
    W2_in=gamma_In*W2_in

    gamma12=0                                       # Multiplicative factor of the connectivity matrix between the 2 ESNs

else:

                                                    # Connected reservoirs (Hierarchical ESNs)
    W1_in=0                                         # Input Connectivity matrix for ESN 1, that is 0, since ESN 1 receives the input from ESN 2 only

    W2_in=np.ones([N2])                             # Input Connectivity matrix for ESN 2, that for a 1-d input we set usually as a random 1,-1 mask
    W2_in[np.random.uniform(0,1,N2)<0.5]=-1
    W2_in=gamma_In*W2_in
    gamma12=1                                       # Multiplicative factor of the connectivity matrix between the 2 ESNs



T=np.shape(X_tr)[1]                                 # Length of the sequence (size of the MNIST image)

# Training Parameters
# With LOW_HS=True, the readout will be defined from 4 hidden states only
# With LOW_HS=False, the readout will be defined from 28 hidden states


N_train=5000                                        # Number of training iterations, for a good performance N_train should be higher and the learning rates should decrease across the simulation time (This simulation is to find the best alphas only)
batch_size=50

LOW_HS=True
if LOW_HS==True:
    T_size=4                                        # Number of hidden states used for the readout. They will be equally spaced across the sequence length...

    alpha_size=0.01                                 # Learning rates for the output weights W_out
    alpha1_Adam=0.001                               # Learning rates for the leakage terms alpha1 and alpha2, they will be given to the manual application of the Adam optimizer
    alpha2_Adam=0.001

else:
    T_size=28                                       # Number of hidden states used for the readout. They will be equally spaced across the sequence length...

    alpha_size=0.001                                # Learning rates for the output weights W_out
    alpha1_Adam=0.001                               # Learning rates for the leakage terms alpha1 and alpha2, they will be given to the manual application of the Adam optimizer
    alpha2_Adam=0.001

beta1=0.999                                         # 1st Order momentum for Adam on the leakage terms
beta2=0.999                                         # 2nd Order momentum for Adam on the leakage terms (Default value)
epsilon=10**(-8)                                    # epsilon for Adam on the leakage terms (Default value)

T_conc=np.int(T/T_size)                             # Number that defines when the states of the NN are used for the readout (the states used are equally spaced, thus it will be sufficient to consider multiples of T_conc)
N_reset=1000                                        # Number of steps at which the output weights are randomly re-initilised (in order to avoid local minima and allow a further optimisation of alpha_1 and alpha_2, see Paper)


# Definition of tensorflow graph
# The placeholders are:
# init_state1, starting states values for the sequence and for ESN 1 (tensor of zeros of Dimensionality [batch_size,N1])
# init_state2, starting states values for the sequence and for ESN 2 (tensor of zeros of Dimensionality [batch_size,N2])
# input, input from the external signal
# prev_state2, previous states values for ESN 2
# prev_alpha11, eligibility trace d x_1 / d alpha_1
# prev_alpha12, eligibility trace d x_1 / d alpha_2
# prev_alpha22, eligibility trace d x_2 / d alpha_2
# prev_m01 (prev_m02) previous value of the running average of the first momentum of the Adam optimizer and for ESN 1 (ESN 2)
# prev_m11, (prev_m12) previous value of the running average of the second momentum of the Adam optimizer and for ESN1 (ESN 2)
# y_true, desired outpuut values

# The Variables subjected to training are:
# W_out, output connectivity
# b, output bias
# alpha1, leakage term of ESN 1
# alpha2, Leakage term of ESN 2

Echo=ECHO_Double(pho1, diluition1, N1, W1_in, pho2, diluition2, N2, W2_in, gamma12, T, T_conc)

init_state1=tf.placeholder(tf.float32,[None,N1])
init_state2=tf.placeholder(tf.float32,[None,N2])

init_alpha11=tf.placeholder(tf.float32,[None,N1])
init_alpha12=tf.placeholder(tf.float32,[None,N1])
init_alpha22=tf.placeholder(tf.float32,[None,N2])

input=tf.placeholder(tf.float32,[None,T])

alpha1=tf.Variable(0.2,dtype=tf.float32,trainable=False)                                                                                                        # Variable for alpha_1, defining the starting value, that can be changed in the interval [0 1]
alpha2=tf.Variable(0.2,dtype=tf.float32,trainable=False)                                                                                                        # Variable for alpha_2, defining the starting value, that can be changed in the interval [0 1]

x, x1, x2, elig_alpha11, elig_alpha12, elig_alpha22=Echo.evolution_graph(init_state1,init_state2,input,alpha1,alpha2,init_alpha11,init_alpha12,init_alpha22)    # Definition of the tensorflow graph for the computation of the values assumed by the nodes across the input


elig_alpha1=tf.concat([elig_alpha11,tf.zeros([tf.shape(elig_alpha11)[0],N2,T_size])],axis=1)                                                                    # Complete eligibility for alpha1, i.e. d x/ d alpha_1, where x=[x_1,x_2]
elig_alpha2=tf.concat([elig_alpha12,elig_alpha22],axis=1)                                                                                                       # Complete eligibility for alpha2, i.e. d x/ d alpha_2, where x=[x_1,x_2]

elig1=tf.reshape(elig_alpha1,[-1,T_size*N])
elig2=tf.reshape(elig_alpha2,[-1,T_size*N])

x=tf.reshape(x,[-1,T_size*N])

y_true=tf.placeholder(tf.float32,[None,N_out])

W_out=tf.Variable(np.random.randn(N*T_size,N_out)/N**2,dtype=tf.float32)
b=tf.Variable(np.ones([N_out])*(-np.log(9)),dtype=tf.float32)


y=tf.matmul(x,W_out)+b                                                                                                                                           # NN output


Adam_opt1=Adam_(alpha1_Adam, beta1, beta2, epsilon)                                                                                                              # Class for Adam on alpha_1
Adam_opt2=Adam_(alpha2_Adam, beta1, beta2, epsilon)                                                                                                              # Class for Adam on alpha_2

alpha1_grad=tf.reduce_mean( ( -y_true*(1-tf.nn.sigmoid(y))+(1-y_true)*tf.nn.sigmoid(y) )*tf.matmul(elig1,W_out) )                                                # Computation of the gradient on alpha_1 (input to Adam)
alpha2_grad=tf.reduce_mean( ( -y_true*(1-tf.nn.sigmoid(y))+(1-y_true)*tf.nn.sigmoid(y) )*tf.matmul(elig2,W_out) )                                                # Computation of the gradient on alpha_2 (input to Adam)

# Adam for alpha_1
prev_m01=tf.placeholder(tf.float32,[])
prev_m11=tf.placeholder(tf.float32,[])

alpha1_grad_A, m01, m11=Adam_opt1.compute_grad(prev_m01,prev_m11,alpha1_grad)                                                                                    # Computation of the change on alpha_1 through Adam
alpha_new1=alpha1+alpha1_grad_A                                                                                                                                  # New value of alpha_1
# Assignment of alpha_1, that is bounded between [alpha_min 1]

alpha_min=0.001
assign_alpha1_grad=tf.assign( alpha1, alpha_new1*( tf.cast( tf.math.logical_and(tf.math.greater_equal(alpha_new1,alpha_min),tf.math.less_equal(alpha_new1,1) ), dtype=tf.float32 ) )+tf.cast(tf.math.greater(alpha_new1,1),dtype=tf.float32)+alpha_min*tf.cast(tf.math.less(alpha_new1,alpha_min),dtype=tf.float32) )

# Adam for alpha_2
prev_m02=tf.placeholder(tf.float32,[])
prev_m12=tf.placeholder(tf.float32,[])

alpha2_grad_A, m02, m12=Adam_opt2.compute_grad(prev_m02,prev_m12,alpha2_grad)                                                                                    # Computation of the change on alpha_2 through Adam
alpha_new2=alpha2+alpha2_grad_A                                                                                                                                  # New value of alpha_2
# Assignment of alpha_2, that is bounded between [alpha_min 1]

assign_alpha2_grad=tf.assign( alpha2, alpha_new2*( tf.cast( tf.math.logical_and(tf.math.greater_equal(alpha_new2,alpha_min),tf.math.less_equal(alpha_new2,1) ), dtype=tf.float32 ) )+tf.cast(tf.math.greater(alpha_new2,1),dtype=tf.float32)+alpha_min*tf.cast(tf.math.less(alpha_new2,alpha_min),dtype=tf.float32) )


error=tf.losses.sigmoid_cross_entropy(y_true,y)
train=tf.train.AdamOptimizer(learning_rate=alpha_size).minimize(error,var_list=[W_out,b])


init=tf.global_variables_initializer()


# Variables that are saved during the simulation
error_save=np.zeros([N_train])

error_save=np.zeros([N_train])
error_mean_save=np.zeros([N_train])

acc_save=np.zeros([N_train])
acc_mean_save=np.zeros([N_train])

alpha1_save=np.zeros([N_train])
alpha1_grad_mean_save=np.zeros([N_train])
alpha1_grad_save=np.zeros([N_train])
alpha2_save=np.zeros([N_train])
alpha2_grad_mean_save=np.zeros([N_train])
alpha2_grad_save=np.zeros([N_train])


m01_save=np.zeros([N_train])
m11_save=np.zeros([N_train])
m02_save=np.zeros([N_train])
m12_save=np.zeros([N_train])

N_mean=200                                                                  # Number of steps for the running average of the quantities saved
N_plot=N_reset                                                              # Number of steps after which we plot the quantities saved during the simulation

with tf.Session() as sess:

    sess.run(init)

    for n in range(N_train):

        if n==0:

            m01_=0
            m11_=0
            m02_=0
            m12_=0


        rand_ind=np.random.randint(0,np.shape(X_tr)[0],(batch_size,))       # Random indices for a sample of input

        images=X_tr[rand_ind,:]

        labels=Y_tr[rand_ind,:]

        y_,error_,m01_,m11_,m02_,m12_,alpha1_,alpha2_,alpha1_grad_,alpha2_grad_,_,_,_=sess.run(
        [y,error,m01,m11,m02,m12,alpha1,alpha2,alpha1_grad,alpha2_grad,assign_alpha1_grad,assign_alpha2_grad,train],
        feed_dict={input:images,init_state1:np.zeros([batch_size,N1]),init_state2:np.zeros([batch_size,N1]),init_alpha11:np.zeros([batch_size,N1]),init_alpha12:np.zeros([batch_size,N1]),init_alpha22:np.zeros([batch_size,N2]),
        prev_m01:m01_,prev_m11:m11_,prev_m02:m02_,prev_m12:m12_,y_true:labels})


        acc=np.mean(np.equal(np.argmax(y_,1),np.argmax(labels,1)))

        alpha1_grad_save[n]=alpha1_grad_
        alpha2_grad_save[n]=alpha2_grad_


        m01_save[n]=m01_
        m11_save[n]=m11_
        m02_save[n]=m02_
        m12_save[n]=m12_

        Adam_opt1.n=Adam_opt1.n+1
        Adam_opt2.n=Adam_opt2.n+1

        error_save[n]=error_

        # Reset of the output weights
        if n%N_reset==0 and n>0:

            print('Reset')
            print(alpha1_,alpha2_)
            print(np.mean(error_save[n-1000:n]))
            sess.run(tf.assign(W_out,tf.random.normal([N*T_size,N_out])/N**2))

            Adam_opt1.n=0
            Adam_opt2.n=0

            m01_=0
            m11_=0
            m02_=0
            m12_=0


        print(n)


        acc_save[n]=acc
        alpha1_save[n]=alpha1_
        alpha2_save[n]=alpha2_

        # Plot of the saved quantities
        if  n%N_plot==0 and n>0:

            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10)) = plt.subplots(5, 2)
            index=np.random.randint(0,batch_size)
            ax1.plot(acc_save[0:n])
            ax1.plot(acc_mean_save[0:n])
            ax2.plot(error_save[0:n])
            ax2.plot(error_mean_save[0:n])

            ax3.plot(alpha1_grad_mean_save[0:n],'b')
            ax4.plot(alpha2_grad_mean_save[0:n],'r')
            ax5.plot(alpha1_grad_save[0:n],'b')
            ax6.plot(alpha2_grad_save[0:n],'r')
            ax7.plot(alpha1_save[0:n],'b')
            ax8.plot(alpha2_save[0:n],'r')
            ax9.plot(m01_save[0:n],'b')
            ax10.plot(m02_save[0:n],'r')
            #plt.show()

        # Computation of the running average
        if n>N_mean:

            error_mean_save[n]=np.mean(error_save[n-N_mean:n])
            acc_mean_save[n]=np.mean(acc_save[n-N_mean:n])
            alpha1_grad_mean_save[n]=np.mean(alpha1_grad_save[n-N_mean:n])
            alpha2_grad_mean_save[n]=np.mean(alpha2_grad_save[n-N_mean:n])

            print(error_mean_save[n])
            print(acc_mean_save[n])
            print('alphas')
            print(alpha1_)
            print(alpha2_)

# Saving the results
error_save=np.expand_dims(error_save,1)
acc_save=np.expand_dims(acc_save,1)
alpha1_save=np.expand_dims(alpha1_save,1)
alpha2_save=np.expand_dims(alpha2_save,1)
alpha1_grad_save=np.expand_dims(alpha1_grad_save,1)
alpha2_grad_save=np.expand_dims(alpha2_grad_save,1)

Results=np.concatenate([error_save,acc_save,alpha1_save,alpha2_save,alpha1_grad_save,alpha2_grad_save],axis=1)


io.savemat('Results_psMNIST_Conn_02_02.mat',{"array": Results})
