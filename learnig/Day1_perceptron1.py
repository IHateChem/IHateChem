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
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("/workspace/IHateChem/deep-learning-from-scratch-master/deep-learning-from-scratch-master/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) 
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
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


'''
0710: 손실함수: 신경망이 지표를 삼아서 탐색하도록 하는 것. 

보통 Sum of Squares for error(sse), croos entropy errror를 자주 이용한다. 

SSE: 1/2 * sigma (y_k - t_k)^2 y_k: 신경망의 출력을 의미 t_k: 정답레이블 k: 차원
CEE: - sigma t_k * log y_k


학습 시킬때 보통 빅데이터로 하기에 전부다 활용하여 학습 시키기엔 현실적인 부담이 있음:

그래서 모집단에서 부분적으로 추출하여 학습을 진행 -> 정규화를 시킴. 

'''




def SSE(y,t): #Sum of square of error
    return 0.5*np.sum((y-t)**2)
def CEE(y,t): #cross Entropy Error
    delta = 1e-7 #0이 들어갔을때 =무한대 나오는거 방지하기 위해 아주 작은 숫자를 더해쥼ㅇㅅㅇ
    return -1*np.sum(t*np.log(y+delta))



