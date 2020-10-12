import tensorflow as tf
import numpy as np



class ECHO_Double:

    def __init__(self, pho1, diluition1, N1, W1_in,
                          pho2, diluition2, N2, W2_in, gamma12, T, T_conc):

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

            self.T=T
            self.T_conc=T_conc


    def evolution_graph(self,init_state1,init_state2,input,alpha1,alpha2,init_alpha11,init_alpha12,init_alpha22):


        # Loop for the computation of the states and the eligibility traces
        state1=init_state1
        states_train1=[]

        state2=init_state2
        states_train2=[]

        elig_alpha11=init_alpha11
        elig_train11=[]

        elig_alpha12=init_alpha12
        elig_train12=[]

        elig_alpha22=init_alpha22
        elig_train22=[]


        for t in range(self.T):

            # states
            prev_state1=tf.identity(state1)
            prev_state2=tf.identity(state2)

            arg1=tf.matmul(prev_state1,self.W1)+tf.matmul(prev_state2,self.W12)+tf.tile(tf.expand_dims(input[:,t],1),[1,self.N1])*self.W1_in
            arg2=tf.matmul(prev_state2,self.W2)+tf.tile(tf.expand_dims(input[:,t],1),[1,self.N2])*self.W2_in


            state1 = (1-alpha1)*prev_state1+alpha1*tf.tanh(arg1)
            state2 = (1-alpha2)*prev_state2+alpha2*tf.tanh(arg2)


            # eligibility traces
            preve_alpha11=tf.identity(elig_alpha11)
            preve_alpha12=tf.identity(elig_alpha12)
            preve_alpha22=tf.identity(elig_alpha22)

            elig_alpha11=(1-alpha1)*preve_alpha11+tf.math.tanh(arg1)+alpha1*(1-tf.math.tanh(arg1)**2)*tf.matmul(preve_alpha11,self.W1)-prev_state1
            elig_alpha22=(1-alpha2)*preve_alpha22+tf.math.tanh(arg2)+alpha2*(1-tf.math.tanh(arg2)**2)*tf.matmul(preve_alpha22,self.W2)-prev_state2
            elig_alpha12=(1-alpha1)*preve_alpha12+alpha1*(1-tf.math.tanh(arg1)**2)*tf.matmul(preve_alpha12,self.W2)+alpha1*(1-tf.math.tanh(arg1)**2)*tf.matmul(preve_alpha22,self.W12)


            # Condition to save the states used for the readout
            if (t+1)%self.T_conc==0:

                states_train1.append(state1)

                states_train2.append(state2)

                elig_train11.append(elig_alpha11)

                elig_train12.append(elig_alpha12)

                elig_train22.append(elig_alpha22)


        states1=tf.concat([tf.expand_dims(s,2) for s in states_train1],2)

        states2=tf.concat([tf.expand_dims(s,2) for s in states_train2],2)

        states=tf.concat([states1,states2],axis=1)

        elig11=tf.concat([tf.expand_dims(s,2) for s in elig_train11],2)

        elig12=tf.concat([tf.expand_dims(s,2) for s in elig_train12],2)

        elig22=tf.concat([tf.expand_dims(s,2) for s in elig_train22],2)


        return states, state1, state2, elig11, elig12, elig22
