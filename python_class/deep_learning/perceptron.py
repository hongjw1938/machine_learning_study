##And 게이트

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1

AND(0, 0)
AND(1, 0)
AND(0, 1)
AND(1, 1) #이 경우만 true(1) 리턴


#bias사용한 가중치 도입

import numpy as np

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
w*x

np.sum(w*x)
np.sum(w*x) + b

def AND_bias(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) +b
    if tmp <= 0:
        return 0
    else:
        return 1


##NAND게이트 구현
def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

##OR게이트 구현
def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


##XOR게이트
#NAND, OR게이트 조합

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y