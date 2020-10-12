import tensorflow as tf
import numpy as np

class ECHO:

    def __init__(self, dt, pho, diluition, N, W_in):

            self.dt=dt

            ## Reservoir 1

            self.N=N

            ## alpha 1
            #self.alpha=tf.constant(alpha,dtype=tf.float32)

            ## W1 def
            W_np=np.random.uniform(-1,1,[N,N])
            D=np.random.uniform(0,1,(N,N))>np.ones((N,N))*diluition
            W_np=W_np*D.astype(int)

            eig=np.linalg.eigvals(W_np)

            self.pho=pho
            W_np=W_np/(np.max(np.absolute(eig)))
            self.W=tf.Variable(W_np,trainable=False,dtype=tf.float32)

            self.eig=eig

            ## Input W
            self.W_in=tf.Variable(W_in,trainable=False,dtype=tf.float32)


    def evolution_graph(self,prev_state,prev_elig,input,alpha):


        arg=tf.matmul(prev_state,self.pho*self.W)+tf.tile(tf.expand_dims(input,1),[1,self.N])*self.W_in
        state_hidden=( (1-alpha)*prev_state+alpha*tf.math.tanh( arg ) )

        batch_size=tf.shape(prev_state)[0]

        elig=(1-alpha)*prev_elig+tf.math.tanh(arg)+alpha*(1-tf.math.tanh(arg)**2)*tf.matmul(prev_elig,self.W)-prev_state


        return state_hidden, elig


class ECHO_Double:

    def __init__(self,dt, pho1, diluition1, N1, W1_in,
                          pho2, diluition2, N2, W2_in, gamma12):

            self.dt=dt

            ## Reservoir 1

            self.N1=N1


            ## W1 def
            W1_np=np.random.uniform(-1,1,[N1,N1])
            D=np.random.uniform(0,1,(N1,N1))>np.ones((N1,N1))*diluition1
            W1_np=W1_np*D.astype(int)

            eig1=np.linalg.eigvals(W1_np)

            W1_np=pho1*W1_np/(np.max(np.absolute(eig1)))
            self.W1=tf.Variable(W1_np,trainable=False,dtype=tf.float32)

            self.eig1=eig1

            ## Input W
            self.W1_in=tf.Variable(W1_in,trainable=False,dtype=tf.float32)

            ## Reservoir 2

            self.N2=N2

            ## W2 def
            W2_np=np.random.uniform(-1,1,[N2,N2])
            D=np.random.uniform(0,1,(N2,N2))>np.ones((N2,N2))*diluition2
            W2_np=W2_np*D.astype(int)

            eig2=np.linalg.eigvals(W2_np)

            W2_np=pho2*W2_np/(np.max(np.absolute(eig2)))
            self.W2=tf.Variable(W2_np,trainable=False,dtype=tf.float32)

            ## Input W
            self.W2_in=tf.Variable(W2_in,trainable=False,dtype=tf.float32)

            ## Reservoir 2 to Reservoir 1

            W12=gamma12*np.random.randn(N2,N2)

            self.W12=tf.Variable(W12,trainable=False,dtype=tf.float32)


    def evolution_graph(self,prev_state1,prev_state2,input,alpha1,alpha2,preve_alpha11,preve_alpha12,preve_alpha22):



        arg1=tf.matmul(prev_state1,self.W1)+tf.matmul(prev_state2,self.W12)+tf.tile(tf.expand_dims(input,1),[1,self.N1])*self.W1_in
        arg2=tf.matmul(prev_state2,self.W2)+tf.tile(tf.expand_dims(input,1),[1,self.N2])*self.W2_in


        state1 = (1-alpha1)*prev_state1+alpha1*tf.tanh(arg1)

        state2 = (1-alpha2)*prev_state2+alpha2*tf.tanh(arg2)

        elig_alpha11=(1-alpha1)*preve_alpha11+tf.math.tanh(arg1)+alpha1*(1-tf.math.tanh(arg1)**2)*tf.matmul(preve_alpha11,self.W1)-prev_state1
        elig_alpha22=(1-alpha2)*preve_alpha22+tf.math.tanh(arg2)+alpha2*(1-tf.math.tanh(arg2)**2)*tf.matmul(preve_alpha22,self.W2)-prev_state2

        elig_alpha12=(1-alpha1)*preve_alpha12+alpha1*(1-tf.math.tanh(arg1)**2)*tf.matmul(preve_alpha12,self.W2)+alpha1*(1-tf.math.tanh(arg1)**2)*tf.matmul(preve_alpha22,self.W12)


        factor=tf.math.abs(elig_alpha11/(tf.math.abs(elig_alpha22)+tf.math.abs(elig_alpha12)))

        states=tf.concat([state1,state2],axis=1)


        return states, state1, state2, elig_alpha11, elig_alpha12, elig_alpha22, factor
