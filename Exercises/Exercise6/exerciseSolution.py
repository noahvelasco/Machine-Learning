# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 15:15:41 2020

@author: npizz
"""
import numpy as np


p_class = np.array([0.33,0.33,0.33])
p_att_given_class = np.array([[0.72,0.21,0.89,0.47,0.64],
                     [0.32,0.82,0.54,0.82,0.17],
                     [0.76,0.65,0.74,0.31,0.75]])


x = np.array([1,1,1,0,0])

p = np.ones(3)
for i in range(3):
    p[i] = p_class[i]
    for j in range(len(x)):
        if x[j] == 1:
            p[i] = p[i]*p_att_given_class[i,j]
        else:
            p[i] = p[i]*(1-p_att_given_class[i,j])
            
print(">>>\n",p)            
print(np.argmax(p))

p = p/np.sum(p)

print(np.sum(p))

p = np.copy(p_att_given_class)
p[:,x==0] = 1-p_att_given_class[:,x==0]
print(">>>\n",p_att_given_class)
print(">>>\n",p)


pd = np.prod(p,axis=1)*p_class
print(">>>\n",pd)
print(np.argmax(pd))

p1 = p_att_given_class*x
print(">>>\n",p1)

p0 = (1-p_att_given_class)*(1-x)
print(">>>\n",p0)

print(">>>\n",p0+p1)

