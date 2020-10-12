import numpy as np
import matplotlib.pyplot as plt

class NARMA:

    def __init__(self,T,alpha,beta,gamma,Delay):

        self.T=T
        self.alpha=alpha
        #self.Delay=Delay
        self.Delay=Delay
        self.beta=0.1/Delay
        self.gamma=gamma


    def narma(self):

        T=self.T

        y = np.zeros([T+1000])
        u = np.random.uniform(0,1,[T+1000])*0.5

        for k in range(self.Delay,self.T+1000):

            y[k] = self.alpha*y[k-1] + self.beta*y[k-1]*np.sum(y[k-self.Delay:k]) + self.gamma*u[k-1]*u[k-self.Delay] + 0.1

        return u[1000:], y[1000:]


    def Training(self,batch_size):

        T=self.T

        y = np.zeros([batch_size,T])
        u = np.random.uniform(0,1,[batch_size,T])*0.5

        for l in range(batch_size):

            u[l,:], y[l,:]=self.narma()

            while np.mean(y[l,:]>2):

                u[l,:], y[l,:]=self.narma()





        return u[:,:], y[:,:]
