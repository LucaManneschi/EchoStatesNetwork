import numpy as np
import matplotlib.pyplot as plt
from Build_Signal import *
from Echo_activity import *
from Adam_manual import *

from scipy import io



# SIGNAL, input and output
# Definition of the input and the desired output, i.e. the NARMA sequence.
# T is the lenght of the signal,
# alpha, gamma, beta, Delay are the parameters defining the NARMA series
# N_out is the dimensionality of the output (1 for this task)
# batch_size is the number of series that the program will define, and that will be used as a batch size for the optimizer
# Thus, X_tr will have the shape [batch_size,T]...

T=10000

alpha=0.3
gamma=1.5
beta=0.5
Delay=5
T_signal=NARMA(T,alpha,beta,gamma,Delay)

batch_size=10
X_tr, Y_tr=T_signal.Training(batch_size)
X_te, Y_te=T_signal.Training(0)
N_out=1


# Select Plot_sig=True to see an example of the input and the desired output

T_plot=1000
Plot_Sig=False
if Plot_Sig==True:

    plt.figure()

    plt.subplot(311)
    plt.plot(X_tr[0,1000:1000+T_plot])
    plt.plot(Y_tr[0,1000:1000+T_plot])
    plt.subplot(312)
    index=np.random.randint(0,batch_size,)
    plt.plot(X_tr[index,1000:1000+T_plot])
    plt.plot(Y_tr[index,1000:1000+T_plot])
    plt.subplot(313)
    index=np.random.randint(0,batch_size,)
    plt.plot(X_tr[index,1000:1000+T_plot])
    plt.plot(Y_tr[index,1000:1000+T_plot])

    plt.show()



# ESN parameters and definition
# Select Connected=True for the hierarchical structure, and Connected=False for the parallel ESNs
# ESN number 1 is the 'deeper' reservoir in the hierarchical structure (in the parallel structure there is no difference between the 2 ESN)
# ESN number 2 is the reservoir that receives the input in the hierarchical structure

N=200                   # Total number of nodes

N1=np.int(N/2)          # Nodes of the first reservoir
N2=np.int(N/2)          # Nodes of the second reservoir

pho1=0.95               # Value of rho for ESN 1
pho2=0.95               # Value of rho for ESN 2
N_av=10                 # Average number of connections for one node in the reservoir
diluition1=1-N_av/N     # Probability of a zero in the connectivity matrix of ESN 1
diluition2=1-N_av/N     # Probability of a zero in the connectivity matrix of ESN 2
gamma_In=0.2            # Multiplicative factor of the input connectivity matrix
gamma_21=1              # Multiplicative factor of the connectivity matrix between the 2 ESNs, in the unconnected cas it will be set to 0

Connected=True
if Connected==False:

                                                # Unconnected reservoirs (Parallel ESNs)
    W1_in=np.ones([N1])                         # Input Connectivity matrix for ESN 1, that for a 1-d input we set usually as a random 1,-1 mask
    W1_in[np.random.uniform(0,1,N1)<0.5]=-1
    W1_in=gamma_In*W1_in

    W2_in=np.ones([N2])                         # Input Connectivity matrix for ESN 2, that for a 1-d input we set usually as a random 1,-1 mask
    W2_in[np.random.uniform(0,1,N2)<0.5]=-1
    W2_in=gamma_In*W2_in

    gamma12=0                                   # Multiplicative factor of the connectivity matrix between the 2 ESNs


else:

                                                # Connected reservoirs (Hierarchical ESNs)
    W1_in=0                                     # Input Connectivity matrix for ESN 1, that is 0, since ESN 1 receives the input from ESN 2 only

    W2_in=np.ones([N2])                         # Input Connectivity matrix for ESN 2, that for a 1-d input we set usually as a random 1,-1 mask
    W2_in[np.random.uniform(0,1,N2)<0.5]=-1
    W2_in=gamma_In*W2_in
    gamma12=0.2                                 # Multiplicative factor of the connectivity matrix between the 2 ESNs


# Training Parameters

N_train=2000000                                 # Number of training iterations

N_checks=20                                     # Number of checks, in which it is possible to plet different information regarding the training

alpha_size=0.0005                               # Learning rates for the output weights W_out
alpha1_Adam=0.0005*0.01                         # Learning rates for the leakage terms alpha1 and alpha2, they will be given to the manual application of the Adam optimizer
alpha2_Adam=0.0005*0.01
beta1=0.99                                      # 1st Order momentum for Adam on the leakage terms
beta2=0.999                                     # 2nd Order momentum for Adam on the leakage terms (Default value)
epsilon=10**(-8)                                # epsilon for Adam on the leakage terms (Default value)
N_reset=200000                                  # Number of steps at which the output weights are randomly re-initilised (in order to avoid local minima and allow a further optimisation of alpha_1 and alpha_2, see Paper)


# Definition of tensorflow graph
# The placeholders are:
# prev_state1, previous states values for ESN 1 (dimensionality [batch_size,N1])
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

Echo=ECHO_Double(dt, pho1, diluition1, N1, W1_in, pho2, diluition2, N2, W2_in, gamma12)                                                                                  # NN definition

prev_state1=tf.placeholder(tf.float32,[None,N1])
prev_state2=tf.placeholder(tf.float32,[None,N2])

preve_alpha11=tf.placeholder(tf.float32,[None,N1])
preve_alpha12=tf.placeholder(tf.float32,[None,N1])
preve_alpha22=tf.placeholder(tf.float32,[None,N2])

input=tf.placeholder(tf.float32,[None])

alpha1=tf.Variable(0.3,dtype=tf.float32,trainable=False)                                                                                                               # Variable for alpha_1, defining the starting value, that can be changed in the interval [0 1]
alpha2=tf.Variable(0.3,dtype=tf.float32,trainable=False)                                                                                                               # Variable for alpha_2, defining the starting value, that can be changed in the interval [0 1]

x, x1, x2, elig_alpha11, elig_alpha12, elig_alpha22, target=Echo.evolution_graph(prev_state1,prev_state2,input,alpha1,alpha2,preve_alpha11,preve_alpha12,preve_alpha22) # Definition of the tensorfloa graph for the computation of the next states

elig_alpha1=tf.concat([elig_alpha11,tf.zeros([batch_size,N2])],axis=1)                                                                                                  # Complete eligibility for alpha1, i.e. d x/ d alpha_1, where x=[x_1,x_2]
elig_alpha2=tf.concat([elig_alpha12,elig_alpha22],axis=1)                                                                                                               # Complete eligibility for alpha2, i.e. d x/ d alpha_2, where x=[x_1,x_2]

y_true=tf.placeholder(tf.float32,[None,N_out])

W_out=tf.Variable(np.random.randn(N,N_out)/N**2,dtype=tf.float32)
b=tf.Variable(np.ones([N_out])*np.mean(Y_tr),dtype=tf.float32)

y=tf.matmul(x,W_out)+b                                                                                                                                                  # NN output

Adam_opt1=Adam_(alpha1_Adam, beta1, beta2, epsilon)                                                                                                                     # Class for Adam on alpha_1
Adam_opt2=Adam_(alpha2_Adam, beta1, beta2, epsilon)                                                                                                                     # Class for Adam on alpha_2

alpha1_grad=tf.reduce_mean( (y-y_true)*tf.matmul(elig_alpha1,W_out) )                                                                                                   # Computation of the gradient on alpha_1 (input to Adam)
alpha2_grad=tf.reduce_mean( (y-y_true)*tf.matmul(elig_alpha2,W_out) )                                                                                                   # Computation of the gradient on alpha_2 (input to Adam)

# Adam for alpha_1
prev_m01=tf.placeholder(tf.float32,[])
prev_m11=tf.placeholder(tf.float32,[])

alpha1_grad_A, m01, m11=Adam_opt1.compute_grad(prev_m01,prev_m11,alpha1_grad)                                                                                           # Computation of the change on alpha_1 through Adam
alpha_new1=alpha1+alpha1_grad_A                                                                                                                                         # New value of alpha_1
# Assignment of alpha_1, that is bounded between [alpha_min 1]

alpha_min=0.001
assign_alpha1_grad=tf.assign( alpha1, alpha_new1*( tf.cast( tf.math.logical_and(tf.math.greater_equal(alpha_new1,alpha_min),tf.math.less_equal(alpha_new1,1) ), dtype=tf.float32 ) )+tf.cast(tf.math.greater(alpha_new1,1),dtype=tf.float32)+alpha_min*tf.cast(tf.math.less(alpha_new1,alpha_min),dtype=tf.float32) )

# Adam for alpha_2
prev_m02=tf.placeholder(tf.float32,[])
prev_m12=tf.placeholder(tf.float32,[])

alpha2_grad_A, m02, m12=Adam_opt2.compute_grad(prev_m02,prev_m12,alpha2_grad)                                                                                           # Computation of the change on alpha_2 through Adam
alpha_new2=alpha2+alpha2_grad_A                                                                                                                                         # New value of alpha_2
# Assignment of alpha_2, that is bounded between [alpha_min 1]

assign_alpha2_grad=tf.assign( alpha2, alpha_new2*( tf.cast( tf.math.logical_and(tf.math.greater_equal(alpha_new2,alpha_min),tf.math.less_equal(alpha_new2,1) ), dtype=tf.float32 ) )+tf.cast(tf.math.greater(alpha_new2,1),dtype=tf.float32)+alpha_min*tf.cast(tf.math.less(alpha_new2,alpha_min),dtype=tf.float32) )

# Error for the training on W_out, handled automatically be tensorflow
error=tf.reduce_mean(tf.square(y_true-y))
train=tf.train.AdamOptimizer(learning_rate=alpha_size).minimize(error)

t_track=0                                                                                                                                                              # It tracks the time of the NARMA sequence (that will be replayed more than once during training)

init=tf.global_variables_initializer()

error_save=np.zeros([N_train])


# Variables that are saved during the simulation
error_save=np.zeros([N_train])
error_mean_save=np.zeros([N_train])

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

y_save=np.zeros([batch_size,N_train])
y_desired_save=np.zeros([batch_size,N_train])

T_switch=0                                      # T_switch defines the starting amount of iteration where only the outup weights are trained, it is set to zero (we believe that a value >0 could be beneficial sometimes)

with tf.Session() as sess:

    sess.run(init)

    for n in range(N_train):

        if n==0:

            m01_=0
            m11_=0
            m02_=0
            m12_=0

            counter=0
            Conv1=False

        if t_track==0:

            x1_=np.zeros([batch_size,N1])
            x2_=np.zeros([batch_size,N1])

            elig_alpha11_=np.zeros([batch_size,N1])
            elig_alpha12_=np.zeros([batch_size,N1])
            elig_alpha22_=np.zeros([batch_size,N2])

        if Conv1==False:

            if  counter>T_switch:

                Conv1=True

        if Conv1==False:

            # Training of W_out only (with T_switch=0 this never happens)

            y_,x_,x1_,x2_,error_,elig_alpha11_,elig_alpha12_,elig_alpha22_,alpha1_,alpha2_,alpha1_grad_,alpha2_grad_,target_,_=sess.run(
            [y,x,x1,x2,error,elig_alpha11,elig_alpha12,elig_alpha22,alpha1,alpha2,alpha1_grad,alpha2_grad,target,train],
            feed_dict={input:X_tr[:,t_track],prev_state1:x1_,prev_state2:x2_,preve_alpha11:elig_alpha11_,preve_alpha12:elig_alpha12_,preve_alpha22:elig_alpha22_,y_true:np.expand_dims(Y_tr[:,t_track],1)})

            alpha1_grad_save[n]=alpha1_grad_
            alpha2_grad_save[n]=alpha2_grad_


            y_desired_save[:,n]=Y_tr[:,t_track]
            y_save[:,n]=y_[:,0]

            counter=counter+1



        if Conv1==True:

            if t_track<200:

                # The algorithm neglects to change the alphas values for the first t_track steps of the NARMA sequence. However, it computes the eligibility traces

                y_,x_,x1_,x2_,error_,elig_alpha11_,elig_alpha12_,elig_alpha22_,target_=sess.run(
                [y,x,x1,x2,error,elig_alpha11,elig_alpha12,elig_alpha22,target],
                feed_dict={input:X_tr[:,t_track],prev_state1:x1_,prev_state2:x2_,preve_alpha11:elig_alpha11_,preve_alpha12:elig_alpha12_,preve_alpha22:elig_alpha22_,y_true:np.expand_dims(Y_tr[:,t_track],1)})

                y_desired_save[:,n]=Y_tr[:,t_track]
                y_save[:,n]=y_[:,0]


            if t_track>200:

                # Simoultaneous training of the Variables

                y_,x_,x1_,x2_,error_,elig_alpha11_,elig_alpha12_,elig_alpha22_,m01_,m11_,m02_,m12_,alpha1_,alpha2_,alpha1_grad_,alpha2_grad_,target_,_,_,_=sess.run(
                [y,x,x1,x2,error,elig_alpha11,elig_alpha12,elig_alpha22,m01,m11,m02,m12,alpha1,alpha2,alpha1_grad,alpha2_grad,target,assign_alpha1_grad,assign_alpha2_grad,train],
                feed_dict={input:X_tr[:,t_track],prev_state1:x1_,prev_state2:x2_,preve_alpha11:elig_alpha11_,preve_alpha12:elig_alpha12_,preve_alpha22:elig_alpha22_,prev_m01:m01_,prev_m11:m11_,prev_m02:m02_,prev_m12:m12_,y_true:np.expand_dims(Y_tr[:,t_track],1)})

                y_desired_save[:,n]=Y_tr[:,t_track]
                y_save[:,n]=y_[:,0]

                alpha1_grad_save[n]=alpha1_grad_
                alpha2_grad_save[n]=alpha2_grad_


                m01_save[n]=m01_
                m11_save[n]=m11_
                m02_save[n]=m02_
                m12_save[n]=m12_

                Adam_opt1.n=Adam_opt1.n+1
                Adam_opt2.n=Adam_opt2.n+1

        # Reset of the output weights
        if n%N_reset==0:

            print('Reset')
            print(alpha1_,alpha2_)
            print(np.mean(error_save[n-1000:n]))
            sess.run(tf.assign(W_out,tf.random.normal([N,N_out])/N**2))

            Adam_opt1.n=0
            Adam_opt2.n=0

            m01_=0
            m11_=0
            m02_=0
            m12_=0

            t_track=0


        t_track=t_track+1
        print(n)

        if t_track==T:

            t_track=0

        error_save[n]=np.sqrt(error_/np.mean(np.var(Y_tr,1)))
        alpha1_save[n]=alpha1_
        alpha2_save[n]=alpha2_

        # Plot of the saved quantities
        if n%50000==0 and n>0:

            fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8), (ax9, ax10), (ax11, ax12)) = plt.subplots(6, 2)
            index=np.random.randint(0,batch_size)
            ax1.plot(y_desired_save[index,0:n])
            ax1.plot(y_save[index,0:n])
            ax3.plot(error_mean_save[0:n])
            ax5.plot(alpha1_grad_mean_save[0:n],'b')
            ax6.plot(alpha2_grad_mean_save[0:n],'r')
            ax7.plot(alpha1_grad_save[0:n],'b')
            ax8.plot(alpha2_grad_save[0:n],'r')
            ax9.plot(alpha1_save[0:n],'b')
            ax10.plot(alpha2_save[0:n],'r')
            ax11.plot(m01_save[0:n],'b')
            ax12.plot(m02_save[0:n],'r')
            #plt.show() # If not commented, it will show a plot of the saved quantities


        # Computation of the running average
        if n>5000:

            error_mean_save[n]=np.mean(error_save[n-5000:n])
            alpha1_grad_mean_save[n]=np.mean(alpha1_grad_save[n-5000:n])
            alpha2_grad_mean_save[n]=np.mean(alpha2_grad_save[n-5000:n])


# Save Results

error_save=np.expand_dims(error_save,1)
alpha1_save=np.expand_dims(alpha1_save,1)
alpha2_save=np.expand_dims(alpha2_save,1)
alpha1_grad_save=np.expand_dims(alpha1_grad_save,1)
alpha2_grad_save=np.expand_dims(alpha2_grad_save,1)

Results=np.concatenate([error_save,alpha1_save,alpha2_save,alpha1_grad_save,alpha2_grad_save],axis=1)


io.savemat('Results_NARMA5_Connected_03_03.mat',{"array": Results})
