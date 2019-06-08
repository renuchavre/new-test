# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 14:02:12 2019

@author: AI & ML
"""

#Plotting the dataset

import numpy as np
import matplotlib.pyplot as plt

n_pts = 500
np.random.seed(0)

Xa =  np.array([np.random.normal(13,2,n_pts),
                np.random.normal(12,2,n_pts)]).T

Xb = np.array([np.random.normal(8,2,n_pts),
                np.random.normal(6,2,n_pts)]).T

X = np.vstack((Xa,Xb))
Y = np.matrix(np.append(np.zeros(n_pts),np.ones(n_pts))).T

plt.scatter(X[:n_pts,0],X[:n_pts,1],color='r')

plt.scatter(X[n_pts:,0],X[n_pts:,1],color='b')
plt.show()


#importing keras functions

import keras
#sequential = linear stack of layers
#in nural network
from keras.models import Sequential 

#import the layer structure class(eg: dense,convolutional etc..)
#in a dense class every node
#in the first layer is connected to 
#every other node in the next layer
from keras.layers import Dense

#import the optimizer
#'Adam' is the standard optimizer
#it has adaptive learning rate

from keras.optimizers import Adam


#defining the model
model = Sequential()

#adding layers to the structure
#by model.add function
#the arg takes properties off the dense layer
#Dense(), 1st arg is the unit=1(because we have only one output node)
#2nd arg in no. of input nodes defined by input_shape
#3rd arg is the activation function

model.add(Dense(units = 1, 
                input_shape = (2,),
                activation='sigmoid'))

#now its time to apply gradient descent
#to slowly move towards the best fit line
#an instance of Adam optimizer 
#with a learning rate of 0.1
adam = Adam(lr = 0.1)

#in order to train the model 
#to classify the data
#we first need to configure the leearning process
#this is done by model.compile function
#inside compile specify:
#the optimizer,error function, cross entropy,
#and the matrix

#error function is defined by loss key argument
#to classify between 2 classes use binary_crossentropy
#to classify between more classes use categorical_crossentropy

model.compile(adam, 
              loss = "binary_crossentropy",
              metrics= ['accuracy'])

#fitting the model on training dataset

history = model.fit(x=X,y=Y,verbose=1,
          batch_size=50,
          validation_split = 0.1,
          epochs=10)

plt.plot(history.history['loss'],color='c')
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title('Loss')
plt.xlabel('epoch')


plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.legend(['training','validation'])
plt.title('Accuracy')
plt.xlabel('epoch')


def plot_decision_boundary(X, y, model):
    x_span = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1)
    y_span = np.linspace(min(X[:,1]) - 1, max(X[:,1]) + 1)
    xx, yy = np.meshgrid(x_span, y_span)
    xx_, yy_ = xx.ravel(), yy.ravel()
    grid = np.c_[xx_, yy_]
    pred_func = model.predict(grid)
    z = pred_func.reshape(xx.shape)
    plt.contourf(xx, yy, z)
plot_decision_boundary(X, Y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

plot_decision_boundary(X, Y, model)
plt.scatter(X[:n_pts,0], X[:n_pts,1])
plt.scatter(X[n_pts:,0], X[n_pts:,1])

x = 7.5
y = 10
 
 
point = np.array([[x, y]])
prediction = model.predict(point)
plt.plot([x], [y], marker='o', markersize=10, color="blue")
print("prediction is: ",prediction)


