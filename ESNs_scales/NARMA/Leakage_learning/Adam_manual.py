import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import io

class Adam_:

    def __init__(self, alpha, beta1, beta2, epsilon):

        self.alpha_A=tf.constant(alpha,dtype=tf.float32)

        self.beta1=tf.constant(beta1,dtype=tf.float32)

        self.beta2=tf.constant(beta2,dtype=tf.float32)

        self.epsilon=tf.constant(epsilon,dtype=tf.float32)

        self.n=1

    def compute_grad(self,prev_m0,prev_m1,grad):

        m0=prev_m0*self.beta1+(1-self.beta1)*grad

        m1=prev_m1*self.beta2+(1-self.beta2)*grad*grad

        m0t=m0/(1-self.beta1**self.n)
        m1t=m1/(1-self.beta2**self.n)

        grad=-self.alpha_A*m0t/(tf.math.sqrt(m1t)+self.epsilon)

        return grad, m0, m1
