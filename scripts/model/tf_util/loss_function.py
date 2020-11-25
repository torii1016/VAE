import tensorflow as tf

def cross_entropy(x, y):
    return tf.reduce_sum(-1.0*x*tf.log(y+1.0e-10) - (1.0-x)*tf.log(1.0-y+1.0e-10), 1)

def KL(mu, log_sigma):
    return -0.5*tf.reduce_sum(1 + 2*log_sigma - mu*mu - tf.exp(2*log_sigma), 1)