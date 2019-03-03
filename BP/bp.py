# _*_coding:utf-8 _*_
import urllib2
import numpy
from numpy import *
import matplotlib.pyplot as plt
def tanh(x):
    return numpy.tanh(x)
def tanh_derivative(x):
    return 1.0 - tanh(x) * tanh(x)
def logistic(x):
    return 1.0 / (1.0 + numpy.exp(-x))
def logistic_derivative(x):
    return logistic(x) * (1.0 - logistic(x) )

def activation(x):
    return tanh(x)
def activation_deriv(x):
    return tanh_derivative(x)

    
def get_init_weights(layers):
    weights =[]
    for i in range(1,len(layers)):
        weights.append( (2*random.random((layers[i-1] + 1, layers[i] + 1 ))-1 ) * 0.25 )
       
    return weights
def get_d_weights(d,yj,weights,layerNum):
    error = yj - d[-1]
    deltas = [error*activation_deriv(d[-1])]
    
    for j in range(layerNum,0,-1):
        
        deltas.append(dot(deltas[-1],weights[j].T)*activation_deriv(d[j]))
    
    return deltas
def fit_weights(x,y,weights,learning_rate = 0.2,epochs = 10000):
    x = c_[x,ones(len(x))]
    
    y = array(y)
    m,n = shape(x)

    for k in range(epochs):
        i = random.randint(m)
        xiout = [x[i]]
        
        for l in range(len(weights)):
            xiout.append(activation(numpy.dot(xiout[l],weights[l])))
        
        error = y[i] -xiout[-1]
        
        deltas = get_d_weights(xiout,y[i],weights,len(xiout)-2)
        deltas.reverse()
        
        for i in range(len(weights)):
            layer_out = atleast_2d(xiout[i])
            delta = atleast_2d(deltas[i])
            weights[i] += learning_rate * dot(layer_out.T,delta)
    return weights
    
def predict(x,weights):
    tx = r_[x,ones(1)]
    for l in range(0,len(weights)):
        tx = activation(dot(tx,weights[l]))
    return tx   
init_weights = get_init_weights([2, 2,2, 1])

x = array([[0,0],[0,1],[1,0],[1,1]])
y = array([0,1,1,0])
weights = fit_weights(x, y,init_weights)
for i in [[0,0],[0,1],[1,0],[1,1]]:
    print (i, predict(i,weights))
