import tensorflow as tf
import numpy as np


class Models:

    def __init__(self,alpha_size,Pns,batch_size,N_ep,N_check,Data_train,Data_val,Data_test,Y_tr,Y_val,Y_te):


        self.Data_train=Data_train
        self.Data_val=Data_val
        self.Data_test=Data_test

        self.Y_tr=Y_tr
        self.Y_val=Y_val
        self.Y_te=Y_te


        self.T=np.shape(Data_train)[2]

        self.N=np.shape(Data_train)[1]

        self.N_class=np.shape(Y_tr)[1]

        self.alpha_size=alpha_size

        self.batch_size=batch_size

        self.N_episodes=N_ep

        self.N_check=N_check

        self.Pns=Pns


    def Standard(self,state,y_true):

        alpha_sizes=self.alpha_size

        N_copies=np.shape(alpha_sizes)[0]

        W_out=[]
        y=[]
        error=[]
        train=[]

        for i in range(N_copies):

            W_out.append(tf.Variable(np.random.uniform(-1,1,[self.N*self.T,self.N_class])/(self.N/10),dtype=tf.float32))

            y.append( tf.matmul(state,W_out[i]) )

            error.append(tf.losses.sigmoid_cross_entropy(y_true,y[i]))

            train.append(tf.train.AdamOptimizer(learning_rate=alpha_sizes[i]).minimize(error[i],var_list=[W_out[i]]))

        return y, error, train

    def SpaRCe(self,state,y_true,theta_g_start):

        alpha_size=self.alpha_size[0]

        N_copies=np.shape(theta_g_start)[0]


        if np.shape(self.alpha_size)[0]>1:

            alpha_size=self.alpha_size[0]

            print('Only one value of learning rate should be specified')

        theta_g=[]
        theta_i=[]
        W_out=[]
        state_sparse=[]
        y=[]
        error=[]
        train1=[]
        train2=[]

        theta_istart=np.random.randn(self.N,self.T)/self.N
        theta_istart=np.reshape(theta_istart,[-1,self.N*self.T])

        for i in range(N_copies):

            theta_g.append(tf.Variable(theta_g_start[i], trainable=False, dtype=tf.float32))

            theta_i.append(tf.Variable(theta_istart,dtype=tf.float32))

            W_out.append(tf.Variable(np.random.uniform(0,1,[self.N*self.T,self.N_class])/(self.N),dtype=tf.float32))

            state_sparse.append(tf.sign(state)*tf.nn.relu(tf.abs(state)-theta_g[i]-theta_i[i]))

            y.append(tf.matmul(state_sparse[i],W_out[i]))

            error.append(tf.losses.sigmoid_cross_entropy(y_true,y[i]))

            train1.append(tf.train.AdamOptimizer(learning_rate=alpha_size).minimize(error[i],var_list=[W_out[i]]))
            train2.append(tf.train.AdamOptimizer(learning_rate=alpha_size/10).minimize(error[i],var_list=[theta_i[i]]))

            train=train1+train2


        return state_sparse, y, error, train


    def Training(self,MODEL,train_divide,test_divide,val_divide):


        N_train=np.shape(self.Data_train)[0]
        N_test=np.shape(self.Data_test)[0]
        if self.Data_val is not(False):
            N_val=np.shape(self.Data_val)[0]

        N_class=np.shape(self.Y_tr)[1]

        N_train_d=int(np.floor(np.shape(self.Y_tr)[0]/train_divide))
        N_test_d=int(np.floor(np.shape(self.Y_te)[0]/test_divide))
        N_val_d=int(np.floor(np.shape(self.Y_te)[0]/val_divide))


        if MODEL==1:

            theta_g_start=[]

            N_copies=np.shape(self.Pns)[0]

            # INITIALISATION
            for n in range(N_copies):

                theta_g_start_help=np.zeros([self.N,self.T])

                for t in range(self.T):

                    state_help=self.Data_train[:,:,t]

                    theta_g_start_help[:,t]=np.percentile(np.abs(state_help),self.Pns[n],0)
                    #theta_g_start_help[:,t]=np.zeros([self.N])

                theta_g_start.append(np.reshape(theta_g_start_help,[-1,self.T*self.N]))

            States_tr=np.reshape(self.Data_train,[-1,self.N*self.T])
            States_te=np.reshape(self.Data_test,[-1,self.N*self.T])

            if self.Data_val is not(False):

                States_val=np.reshape(self.Data_val,[-1,self.N*self.T])
                N_val_cl=np.sum(self.N*np.sum(np.sum(self.Data_val!=0,1)!=0,1))

            s=tf.placeholder(tf.float32,[None,self.N*self.T])
            y_true=tf.placeholder(tf.float32,[None,self.N_class])

            state_sparse, y, error, train = self.SpaRCe(s,y_true,theta_g_start)

            N_train_cl=np.sum(self.N*np.sum(np.sum(self.Data_train!=0,1)!=0,1))
            N_test_cl=np.sum(self.N*np.sum(np.sum(self.Data_test!=0,1)!=0,1))

            Results_tr=np.zeros([N_copies,self.N_check,3])
            Results_te=np.zeros([N_copies,self.N_check,3])
            Results_val=np.zeros([N_copies,self.N_check,3])

        if MODEL==2:

            States_tr=np.reshape(self.Data_train,[-1,self.N*self.T])
            States_te=np.reshape(self.Data_test,[-1,self.N*self.T])

            if self.Data_val is not(False):

                States_val=np.reshape(self.Data_val,[-1,self.N*self.T])

            s=tf.placeholder(tf.float32,[None,self.N*self.T])
            y_true=tf.placeholder(tf.float32,[None,self.N_class])

            y, error, train = self.Standard(s,y_true)

            N_copies=np.shape(self.alpha_size)[0]

            Results_tr=np.zeros([N_copies,self.N_check,2])
            Results_te=np.zeros([N_copies,self.N_check,2])
            Results_val=np.zeros([N_copies,self.N_check,2])


        init=tf.global_variables_initializer()

        with tf.Session() as sess:

            sess.run(init)

            for n in range(self.N_episodes):

                if n>0:

                    rand_ind=np.random.randint(0,N_train,(self.batch_size,))

                    images=States_tr[rand_ind,:]

                    labels=self.Y_tr[rand_ind,:]

                    _=sess.run([t for t in train],feed_dict={y_true:labels,s:images})


                if n%(np.round(self.N_episodes/self.N_check))==0:

                    index_help=int(n/(np.round(self.N_episodes/self.N_check)))

                    for i in range(N_copies):

                        matches=tf.equal(tf.argmax(y_true,1),tf.argmax(y[i],1))
                        p=tf.reduce_mean(tf.cast(matches,tf.float32))

                        if MODEL==2:

                            p_test=0
                            error_test=0

                            for l in range(test_divide):

                                images=States_te[l*N_test_d:(l+1)*N_test_d,:]
                                labels=self.Y_te[l*N_test_d:(l+1)*N_test_d,:]

                                p_test_,error_test_=sess.run([p,error[i]],feed_dict={y_true:labels,s:images})

                                p_test=p_test+p_test_*N_test_d/N_test
                                error_test=error_test+error_test*N_test_d/N_test

                            Size=N_test-test_divide*N_test_d

                            if Size>0:

                                images=States_te[test_divide*N_test_d:,:]
                                labels=self.Y_te[test_divide*N_test_d:,:]

                                p_test_,error_test_=sess.run([p,error[i]],feed_dict={y_true:labels,s:images})

                                p_test=p_test+p_test_*Size/N_test
                                error_test=error_test+error_test*Size/N_test


                            print('Test set, Iteration: ',n,'ESN ',i,'Probability: ',p_test,'Error: ',error_test)

                            Results_te[i,index_help,0]=error_test
                            Results_te[i,index_help,1]=p_test


                            p_train=0
                            error_train=0

                            for l in range(train_divide):

                                images=States_tr[l*N_train_d:(l+1)*N_train_d,:]
                                labels=self.Y_tr[l*N_train_d:(l+1)*N_train_d,:]

                                p_train_,error_train_=sess.run([p,error[i]],feed_dict={y_true:labels,s:images})

                                p_train=p_train+p_train_*N_train_d/N_train
                                error_train=error_train+error_train_*N_train_d/N_train

                            Size=N_train-train_divide*N_train_d

                            if Size>0:

                                images=States_tr[train_divide*N_train_d:,:]
                                labels=self.Y_tr[train_divide*N_train_d:,:]

                                p_train_,error_train_=sess.run([p,error[i]],feed_dict={y_true:labels,s:images})

                                p_train=p_train+p_train_*Size/N_train
                                error_train=error_train+error_train_*Size/N_train


                            print('Training set, Iteration: ',n,'ESN ',i,'Probability: ',p_train,'Error: ',error_train)


                            Results_tr[i,index_help,0]=error_train
                            Results_tr[i,index_help,1]=p_train


                            p_val=0
                            error_val=0

                            if self.Data_val is not(False):

                                for l in range(val_divide):

                                    images=States_val[l*N_val_d:(l+1)*N_val_d,:]
                                    labels=self.Y_val[l*N_val_d:(l+1)*N_val_d,:]

                                    p_val_,error_val_=sess.run([p,error[i]],feed_dict={y_true:labels,s:images})

                                    p_val=p_val+p_val_*N_val_d/N_val
                                    error_val=error_val+error_val_*N_val_d/N_val

                                Size=N_val-val_divide*N_val_d

                                if Size>0:

                                    images=States_val[val_divide*N_val_d:,:]
                                    labels=self.Y_val[val_divide*N_val_d:,:]

                                    p_val_,error_val_=sess.run([p,error[i]],feed_dict={y_true:labels,s:images})

                                    p_val=p_val+p_val_*Size/N_val
                                    error_val=error_val+error_val_*Size/N_val


                                print('Validation set, Iteration: ',n,'ESN ',i,'Probability: ',p_val,'Error: ',error_val)

                                Results_val[i,index_help,0]=error_val
                                Results_val[i,index_help,1]=p_val




                        if MODEL==1:

                            p_test=0
                            error_test=0
                            cl=0

                            for l in range(test_divide):

                                images=States_te[l*N_test_d:(l+1)*N_test_d,:]
                                labels=self.Y_te[l*N_test_d:(l+1)*N_test_d,:]

                                state_,p_test_,error_test_=sess.run([state_sparse[i],p,error[i]],feed_dict={y_true:labels,s:images})
                                cl_=np.sum(state_!=0)/N_test_cl

                                p_test=p_test+p_test_/test_divide
                                error_test=error_test+error_test_/test_divide
                                cl=cl+cl_

                            Size=N_test-test_divide*N_test_d

                            if Size>0:

                                images=States_te[test_divide*N_test_d:,:]
                                labels=self.Y_te[test_divide*N_test_d:,:]

                                state_,p_test_,error_test_=sess.run([state_sparse[i],p,error[i]],feed_dict={y_true:labels,s:images})
                                cl_=np.sum(state_!=0)/N_test_cl

                                p_test=p_test+p_test_*Size/N_test
                                error_test=error_test+error_test*Size/N_test
                                cl=cl+cl_

                            print('Test set, Iteration: ',n,'SpaRCe ',i,'Probability: ',p_test,'Error: ',error_test,'Coding ',cl)

                            Results_te[i,index_help,0]=error_test
                            Results_te[i,index_help,1]=p_test
                            Results_te[i,index_help,2]=cl

                            p_train=0
                            error_train=0
                            cl=0

                            for l in range(train_divide):

                                images=States_tr[l*N_train_d:(l+1)*N_train_d,:]
                                labels=self.Y_tr[l*N_train_d:(l+1)*N_train_d,:]

                                state_,p_train_,error_train_=sess.run([state_sparse[i],p,error[i]],feed_dict={y_true:labels,s:images})
                                cl_=np.sum(state_!=0)/N_train_cl

                                p_train=p_train+p_train_/train_divide
                                error_train=error_train+error_train_/train_divide
                                cl=cl+cl_

                            Size=N_train-train_divide*N_train_d

                            if Size>0:

                                images=States_tr[train_divide*N_train_d:,:]
                                labels=self.Y_tr[train_divide*N_train_d:,:]

                                state_,p_train_,error_train_=sess.run([state_sparse[i],p,error[i]],feed_dict={y_true:labels,s:images})
                                cl_=np.sum(state_!=0)/N_train_cl

                                p_train=p_train+p_train_*Size/N_train
                                error_train=error_train+error_train_*Size/N_train
                                cl=cl+cl_

                            print('Training set, Iteration: ',n,'SpaRCe ',i,'Probability: ',p_train,'Error: ',error_train,'Coding ',cl)

                            Results_tr[i,index_help,0]=error_train
                            Results_tr[i,index_help,1]=p_train
                            Results_tr[i,index_help,2]=cl


                            if self.Data_val is not(False):

                                p_val=0
                                error_val=0
                                cl=0

                                for l in range(val_divide):

                                    images=States_val[l*N_val_d:(l+1)*N_val_d,:]
                                    labels=self.Y_val[l*N_val_d:(l+1)*N_val_d,:]

                                    state_,p_val_,error_val_=sess.run([state_sparse[i],p,error[i]],feed_dict={y_true:labels,s:images})
                                    cl_=np.sum(state_!=0)/N_val_cl

                                    p_val=p_val+p_val_/val_divide
                                    error_val=error_val+error_val_/val_divide
                                    cl=cl+cl_

                                Size=N_val-val_divide*N_val_d

                                if Size>0:

                                    images=States_val[val_divide*N_val_d:,:]
                                    labels=self.Y_val[val_divide*N_val_d:,:]

                                    state_,p_val_,error_val_=sess.run([state_sparse[i],p,error[i]],feed_dict={y_true:labels,s:images})
                                    cl_=np.sum(state_!=0)/N_val_cl

                                    p_val=p_val+p_val_*Size/N_val
                                    error_val=error_val+error_val_*Size/N_val

                                print('Validation set, Iteration: ',n,'SpaRCe ',i,'Probability: ',p_val,'Error: ',error_val,'Coding ',cl)

                                Results_val[i,index_help,0]=error_val
                                Results_val[i,index_help,1]=p_val
                                Results_val[i,index_help,2]=cl

        return Results_tr, Results_te, Results_val
