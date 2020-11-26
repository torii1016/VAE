import tensorflow as tf

def cross_entropy(x, y): # Σ_{i=1}^D xi*log(yi)+(1-xi)*log(1-yi)
    return tf.reduce_sum(1.0*x*tf.log(y+1.0e-10) + (1.0-x)*tf.log(1.0-y+1.0e-10), 1)

def KL(mu, log_sigma): # -0.5*Σ_{j=1}^J(1+log(σ_j^2)-μ_j^2-σ_j^2)
    return -0.5*tf.reduce_sum(1 + 2*log_sigma - mu*mu - tf.exp(2*log_sigma), 1)