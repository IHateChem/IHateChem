import numpy as np
import matplotlib.pyplot as plt

def AND(x1, x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else: 
        return 1
def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else: 
        return 1
def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = - 0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else: 
        return 1
def test(fun):
    print(f"test function", fun.__name__)
    print(f"it's result of 0,0 : {fun(0,0)} ")
    
    print(f"it's result of 0,1 : {fun(0,1)}")
    print(f"it's result of 1,0 : {fun(1,0)}")
          
    print(f"it's result of 1,1 : {fun(1,1)}")
    print(f"---------------end test----------------")
def XOR(x1,x2):
    y1 = NAND(x1,x2)
    y2 = OR(x1,x2)
    return AND(y1,y2)


#test(XOR)
#
    
    
    
    
    
#todo: abs()
'''
1 1 0
1 0 1
0 1 1
0 0 0


and 1   or 1   0
    0      1   1
    0      1   1 
    0      0   1
'''


'''
-----------------------------0704 공부 내용 시작--------------------------------------

활성화 함수 h(X) -> 기존 퍼셉트론을 한번더 처리를 해주는듯? , 퍼셉트론에서 신경망으로 가는 이정표!
'''


def stepfunction(x):
    y = x>0
    return y.astype(np.int)

'''
0709 딥러닝 - mnist 사용법 익히고 다운로드 받음

'''

import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = \
load_mnist(flatten = True, normalize = False)

img = x_train[0]
label = t_train[0]
print(label)
print(img.shape)
img = img.reshape(28,28)
print(img.shape)
img_show(img)

'''print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)'''