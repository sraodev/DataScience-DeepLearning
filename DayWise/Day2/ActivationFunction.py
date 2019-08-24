# -*- coding: utf-8 -*-
"""
Activation Functions
"""

import numpy as np
import matplotlib.pylab as plt

def step(x):
    return np.array(x > 0, dtype=np.int)
















def sigmoid(x):
    return 1 / (1 + np.exp(-x))  # 0 & 1

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return (np.exp(x)-np.exp(-x)) / (np.exp(x) + np.exp(-x))
                                  # -1 & 1

x = np.arange(-5.0, 5.0, 0.1)

y_step = step(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

fig, axes = plt.subplots(ncols=4, figsize=(20, 5))

#ax = axes[0]
#ax.plot(x, ____)
#ax.set_ylim([-0.5, 1.5])
#ax.set_xlim([-5,5])
#ax.grid(True)
#ax.set_xlabel('x')
#ax.set_title('_______')

x = np.arange(-5.0, 5.0, 0.1)
y_step = step(x)
y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)
#fig, axes = plt.subplots(ncols=4, figsize=(20, 5))

ax = axes[0]
ax.plot(x, y_step)
ax.set_ylim([-0.5, 1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('x')
ax.set_title('Binary Threshold')

ax = axes[1]
ax.plot(x, y_sigmoid)
ax.set_ylim([-0.5, 1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('x')
ax.set_title('Sigmoid function')

ax = axes[2]
ax.plot(x, y_tanh)
ax.set_ylim(-1.,1)
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('x')
ax.set_title('Tanh')

ax = axes[3]
ax.plot(x, y_relu)
ax.set_ylim([-0.5, 1.5])
ax.set_xlim([-5,5])
ax.grid(True)
ax.set_xlabel('x')
ax.set_title('ReLu')
plt.show()
