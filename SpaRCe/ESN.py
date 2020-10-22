import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



class ESN2:

    def __init__(self, alpha1, pho1, diluition1, N1, W1_in,
                       alpha2, pho2, diluition2, N2, W2_in, gamma12, T, T_conc):

            ## Reservoir 1

            self.N1=N1

            self.alpha1=tf.Variable(alpha1,trainable=False,dtype=tf.float32)

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

            self.alpha2=tf.Variable(alpha2,trainable=False,dtype=tf.float32)

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

            W12=gamma12*np.random.randn(N2,N1)

            self.W12=tf.Variable(W12,trainable=False,dtype=tf.float32)

            self.T=T
            #self.T_size=np.int32(self.T/T_conc)

            self.T_size=np.int32(np.shape(T_conc)[0])
            self.T_conc=T_conc

            self.N_in=np.shape(W2_in)[0]
            self.D=np.shape(np.shape(W2_in))[0]


    def evolution_graph(self,init_state1,init_state2,input):


        # Loop for the computation of the states and the eligibility traces
        state1=init_state1
        states_train1=[]

        state2=init_state2
        states_train2=[]

        counter_conc=0

        for t in range(self.T):

            # states
            prev_state1=tf.identity(state1)
            prev_state2=tf.identity(state2)

            if self.D==1:

                arg1=tf.matmul(prev_state1,self.W1)+tf.matmul(prev_state2,self.W12)+tf.tile(tf.expand_dims(input[:,t],1),[1,self.N1])*self.W1_in
                arg2=tf.matmul(prev_state2,self.W2)+tf.tile(tf.expand_dims(input[:,t],1),[1,self.N2])*self.W2_in

            else:

                arg1=tf.matmul(prev_state1,self.W1)+tf.matmul(prev_state2,self.W12)+tf.matmul(input[:,:,t],self.W1_in)
                arg2=tf.matmul(prev_state2,self.W2)+tf.matmul(input[:,:,t],self.W2_in)


            state1 = (1-self.alpha1)*prev_state1+self.alpha1*tf.tanh(arg1)
            state2 = (1-self.alpha2)*prev_state2+self.alpha2*tf.tanh(arg2)


            # Condition to save the states used for the readout
            #if (t+1)%self.T_conc==0:
            if t==self.T_conc[counter_conc]:

                states_train1.append(state1)

                states_train2.append(state2)

                counter_conc=counter_conc+1


        states1=tf.concat([tf.expand_dims(s,2) for s in states_train1],2)

        states2=tf.concat([tf.expand_dims(s,2) for s in states_train2],2)

        states=tf.concat([states1,states2],axis=1)


        return states, state1, state2


    def Computation(self,X_tr,X_te,X_val,train_divide,test_divide,val_divide):

        if self.D==1:

            s=tf.placeholder(tf.float32,[None,self.T])

        if self.D>1:

            s=tf.placeholder(tf.float32,[None,self.N_in,self.T])


        init_state1=tf.placeholder(tf.float32,[None,self.N1])
        init_state2=tf.placeholder(tf.float32,[None,self.N2])

        states, x1, x2=self.evolution_graph(init_state1,init_state2,s)    # Definition of the tensorflow graph for the computation of the values assumed by the nodes across the input

        N=self.N1+self.N2

        N_train_d=int(np.floor(np.shape(X_tr)[0]/train_divide))
        Data_train=np.zeros([np.shape(X_tr)[0],N,self.T_size])

        N_test_d=int(np.floor(np.shape(X_te)[0]/test_divide))
        Data_test=np.zeros([np.shape(X_te)[0],N,self.T_size])

        if X_val is not(False):

            N_val_d=int(np.floor(np.shape(X_val)[0]/val_divide))
            Data_val=np.zeros([np.shape(X_val)[0],self.N,self.T_size])

        init=tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)


            print('Computing States for the training set')

            for l in range(train_divide):

                if self.D>1:
                    images=X_tr[l*N_train_d:(l+1)*N_train_d,:,:]

                else:
                    images=X_tr[l*N_train_d:(l+1)*N_train_d,:]

                states_=sess.run(states,feed_dict={init_state1:np.zeros([N_train_d,self.N1]),init_state2:np.zeros([N_train_d,self.N2]),s:images})

                Data_train[l*N_train_d:(l+1)*N_train_d,:,:]=states_

            Size=np.shape(X_tr)[0]-train_divide*N_train_d

            if Size>0:

                if self.D>1:
                    images=X_tr[train_divide*N_train_d:(l+1)*N_train_d,:,:]

                else:
                    images=X_tr[train_divide*N_train_d:(l+1)*N_train_d,:]


                states_=sess.run(states,feed_dict={init_state1:np.zeros([Size,self.N1]),init_state2:np.zeros([Size,self.N2]),s:images})

                Data_train[train_divide*N_train_d:,:,:]=states_


            print('Computing States for the test set')

            for l in range(test_divide):

                if self.D>1:
                    images=X_te[l*N_test_d:(l+1)*N_test_d,:,:]

                else:
                    images=X_te[l*N_test_d:(l+1)*N_test_d,:]

                states_=sess.run(states,feed_dict={init_state1:np.zeros([N_test_d,self.N1]),init_state2:np.zeros([N_test_d,self.N2]),s:images})

                Data_test[l*N_test_d:(l+1)*N_test_d,:,:]=states_

            Size=np.shape(X_te)[0]-test_divide*N_test_d

            if Size>0:

                if self.D>1:
                    images=X_te[test_divide*N_test_d:(l+1)*N_test_d,:,:]

                else:
                    images=X_te[test_divide*N_test_d:(l+1)*N_test_d,:]

                states_=sess.run(states,feed_dict={init_state1:np.zeros([Size,self.N1]),init_state2:np.zeros([Size,self.N2]),s:images})

                Data_test[test_divide*N_test_d:,:,:]=states_


            if X_val is not(False):

                print('Computing States for the validation set')

                for l in range(val_divide):

                    if self.D>1:
                        images=X_val[l*N_val_d:(l+1)*N_val_d,:,:]

                    else:
                        images=X_val[l*N_val_d:(l+1)*N_val_d,:]

                    states_=sess.run(states,feed_dict={init_state1:np.zeros([N_val_d,self.N1]),init_state2:np.zeros([N_val_d,self.N2]),s:images})

                    Data_val[l*N_val_d:(l+1)*N_val_d,:,:]=states_

                Size=np.shape(X_val)[0]-val_divide*N_val_d

                if Size>0:

                    if self.D>1:
                        images=X_val[val_divide*N_val_d:(l+1)*N_val_d,:,:]

                    else:
                        images=X_val[val_divide*N_val_d:(l+1)*N_val_d,:]


                    states_=sess.run(states,feed_dict={init_state1:np.zeros([Size,self.N1]),init_state2:np.zeros([Size,self.N2]),s:images})

                    Data_val[val_divide*N_val_d:,:,:]=states_

            else:

                Data_val=[]


        show_example=True

        if show_example:

            plt.subplot(2,1,1)
            for i in range(self.N1):
                plt.plot(Data_train[0,i,0:np.shape(states_)[2]-1])


            plt.subplot(2,1,2)
            for i in range(self.N2):
                plt.plot(Data_train[0,self.N1+i,0:np.shape(states_)[2]-1])



        plt.show()


        return Data_train, Data_test, Data_val





class ESN1:

    def __init__(self, alpha, pho, diluition, N, W_in, T, T_conc):

        ## Reservoir 1
        self.N=N
        self.alpha=alpha

        ## alpha
        self.alpha=tf.Variable(alpha, trainable=False, dtype=tf.float32)


        ## W def
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

        self.T=T
        self.T_size=np.int32(np.shape(T_conc)[0])
        self.T_conc=T_conc

        self.N_in=np.shape(W_in)[0]
        self.D=np.shape(np.shape(W_in))[0]


    def evolution_graph(self,init_state,input):


        # Loop for the computation of the states
        state=init_state
        states_train=[]

        counter_conc=0

        for t in range(self.T):

            prev_state=tf.identity(state)

            if self.D==1:

                arg=tf.matmul(prev_state,self.pho*self.W)+tf.tile(tf.expand_dims(input[:,t],1),[1,self.N])*self.W_in

            else:

                arg=tf.matmul(prev_state,self.pho*self.W)+tf.matmul(input[:,:,t],self.W_in)


            state=( (1-self.alpha)*prev_state+self.alpha*tf.math.tanh( arg ) )

            #if (t+1)%self.T_conc==0:
            if t==self.T_conc[counter_conc]:

                states_train.append(state)

                counter_conc=counter_conc+1

        states=tf.concat([tf.expand_dims(s,2) for s in states_train],2)

        return states


    def Computation(self,X_tr,X_te,X_val,train_divide,test_divide,val_divide):

        if self.D==1:

            s=tf.placeholder(tf.float32,[None,self.T])

        if self.D>1:

            s=tf.placeholder(tf.float32,[None,self.N_in,self.T])


        init_state=tf.placeholder(tf.float32,[None,self.N])

        states=self.evolution_graph(init_state,s)                           # Definition of the tensorflow graph for the computation of the values assumed by the nodes across the input

        N_train_d=int(np.floor(np.shape(X_tr)[0]/train_divide))
        Data_train=np.zeros([np.shape(X_tr)[0],self.N,self.T_size])

        N_test_d=int(np.floor(np.shape(X_te)[0]/test_divide))
        Data_test=np.zeros([np.shape(X_te)[0],self.N,self.T_size])

        if X_val is not(False):

            N_val_d=int(np.floor(np.shape(X_val)[0]/val_divide))
            Data_val=np.zeros([np.shape(X_val)[0],self.N,self.T_size])

        init=tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)


            print('Computing States for the training set')

            for l in range(train_divide):

                if self.D>1:
                    images=X_tr[l*N_train_d:(l+1)*N_train_d,:,:]

                else:
                    images=X_tr[l*N_train_d:(l+1)*N_train_d,:]

                states_=sess.run(states,feed_dict={init_state:np.zeros([N_train_d,self.N]),s:images})

                Data_train[l*N_train_d:(l+1)*N_train_d,:,:]=states_

            Size=np.shape(X_tr)[0]-train_divide*N_train_d

            if Size>0:

                if self.D>1:
                    images=X_tr[train_divide*N_train_d:(l+1)*N_train_d,:,:]

                else:
                    images=X_tr[train_divide*N_train_d:(l+1)*N_train_d,:]


                states_=sess.run(states,feed_dict={init_state:np.zeros([Size,self.N]),s:images})

                Data_train[train_divide*N_train_d:,:,:]=states_


            print('Computing States for the test set')

            for l in range(test_divide):

                if self.D>1:
                    images=X_te[l*N_test_d:(l+1)*N_test_d,:,:]

                else:
                    images=X_te[l*N_test_d:(l+1)*N_test_d,:]

                states_=sess.run(states,feed_dict={init_state:np.zeros([N_test_d,self.N]),s:images})

                Data_test[l*N_test_d:(l+1)*N_test_d,:,:]=states_

            Size=np.shape(X_te)[0]-test_divide*N_test_d

            if Size>0:

                if self.D>1:
                    images=X_te[test_divide*N_test_d:(l+1)*N_test_d,:,:]

                else:
                    images=X_te[test_divide*N_test_d:(l+1)*N_test_d,:]

                states_=sess.run(states,feed_dict={init_state:np.zeros([Size,self.N]),s:images})

                Data_test[test_divide*N_test_d:,:,:]=states_


            if X_val is not(False):

                print('Computing States for the validation set')

                for l in range(val_divide):

                    if self.D>1:
                        images=X_val[l*N_val_d:(l+1)*N_val_d,:,:]

                    else:
                        images=X_val[l*N_val_d:(l+1)*N_val_d,:]

                    states_=sess.run(states,feed_dict={init_state:np.zeros([N_val_d,self.N]),s:images})

                    Data_val[l*N_val_d:(l+1)*N_val_d,:,:]=states_

                Size=np.shape(X_val)[0]-val_divide*N_val_d

                if Size>0:

                    if self.D>1:
                        images=X_val[val_divide*N_val_d:(l+1)*N_val_d,:,:]

                    else:
                        images=X_val[val_divide*N_val_d:(l+1)*N_val_d,:]


                    states_=sess.run(states,feed_dict={init_state:np.zeros([Size,self.N]),s:images})

                    Data_val[val_divide*N_val_d:,:,:]=states_

            else:

                Data_val=[]


        show_example=True

        if show_example:

            plt.subplot(1,1,1)
            for i in range(self.N):
                plt.plot(Data_train[0,i,0:np.shape(states_)[2]-1])


        plt.show()


        return Data_train, Data_test, Data_val
