from __future__ import print_function

# gym boilerplate
import numpy as np
import gym
from gym import wrappers
from gym.spaces import Discrete, Box

# keras boilerplate: the simplest way to neural networking
from keras.models import *
from keras.layers import *
from keras.optimizers import *
import keras
from math import *
import random
import keras.backend as K
import time


# model1
inp1 = Input(shape=(2,))
i1 = inp1
i1 = Dense(2)(i1)

m1 = Model(input=inp1,output=i1)
# m1.compile(loss='mse',optimizer='adam')

# for l in m1.layers: l.trainable = False

m1f = Model(input=inp1,output=i1)

m1f.trainable = False

# model2
inp2 = Input(shape=(2,))
i2 = m1f(inp2)
i2 = Dense(2)(i2)

m2 = Model(input=inp2,output=i2)

# model3
inp31,inp32 = Input(shape=(2,)),Input(shape=(2,))
i31 = m2(inp31)
i32 = m2(inp32)
i3 = merge([i31,i32],mode='mul')

m3 = Model(input=[inp31,inp32],output=i3)

# train model3
m3.compile(loss='mse',optimizer='adam')

td1 = np.random.normal(loc=1.,scale=1.,size=(64,2))
td2 = np.random.normal(loc=-1.,scale=1.,size=(64,2))
t3 = (td1+td2) ** 2

print('before train:')
print(m1.get_weights())
print(m2.get_weights())

print('train:')

m3.fit([td1,td2],t3,
batch_size=64,
nb_epoch=10,
verbose=2,
shuffle=False,
)

print('after train:')
print(m1.get_weights())
print(m2.get_weights())
