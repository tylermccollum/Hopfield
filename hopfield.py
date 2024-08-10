#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from PIL import Image


# In[108]:


def plot (arr_in):
    plt.imshow(arr_in, 'Greys')
    plt.show()


# In[109]:


class neuron ():
    def __init__(self, size) -> None:
        self.weights = np.random.rand(size)
        self.output = np.random.random(1)
        pass

    def __repr__(self) -> str:
        return f"Weights = {self.weights} \n            Current state = {self.output}"


first_neuron = neuron(10)
print(first_neuron)


# In[110]:


class hopfield (): 
    def __init__(self, shape) -> None:
        # Set weights to range [-1,1)
        
        self.input_weights = np.identity(shape[1]) * 1.1#np.full((shape[0], shape[1]), 1.0)
        print(self.input_weights)
        self.internal_weights = np.random.rand(shape[0],shape[1])*2 -1 
        #Avoid self-feedback
        np.fill_diagonal(self.internal_weights, 0)
        self.outputs = np.random.rand(shape[0],shape[1])
        self.learning_rate = 0.1

    def __repr__(self) -> str:
        return f"Internal weights: {self.internal_weights}\n\n\
        Current state: {self.outputs}"

    def update (self, input):
        # Behold! ReLU
        # print(f"Input shape: {input.shape}")
        # print(f"Input weights shape: {self.input_weights.shape}")
        input_activation = input * self.input_weights
        # print("input activation")
        # plot(input_activation)
        # print(f"Input activation : {input_activation.shape}")

        # print(f"Internal weights shape: {self.internal_weights.shape}")
        # print(f"Ouputs shape: {self.outputs.shape}")
        self_activation = self.internal_weights @ self.outputs 
        print("self activation")
        plot(self_activation)
        # print(f"Self activation: {self_activation.shape}")

        new_outputs =  input_activation + self_activation
        # print(f"New outputs: {new_outputs}")
        new_outputs = np.clip(new_outputs, 0, None)
        print("new outputs")
        plot(new_outputs)
        # print( new_outputs )

        hebb = new_outputs @ self.outputs #self.outputs @ new_outputs.T#
        # print(hebb)
        self.internal_weights = np.multiply( hebb, self.learning_rate)
        np.fill_diagonal(self.internal_weights, 0)

    def visualize_weights (self):
        plt.imshow(self.internal_weights, 'Greys')

    def visualize_outputs (self):
        plt.imshow(self.outputs, 'Greys')

hop = hopfield([10,10])
print(hop)
hop.visualize_weights()


# In[111]:


smily = plt.imread('smily.png')
smily = 1 -smily
smily = np.array(smily).astype('float')
print(smily )
plt.imshow(smily, 'Greys')


# In[112]:


for i in range (0, 100):
    hop.update(smily)
    hop.visualize_weights()
    hop.visualize_outputs()
    
    


# In[ ]:




