import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline 
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
시그모이드아 계단 함수의 차이: 연속적인 값 반환! 그러나 크게 보면 둘이비슷함(수렴)
공통점: 비선형 함수 

과거에는 시그모이드 많이 사용했으나
최근에는 ReLu많이 사용. -> 0이하면0 출력, 그이상이면 그대로 출력:
h(x) = x (x>0) , 0(x<=0)

기계학습은 분류와 회귀로 나뉨. 분류는 어디에 속하느냐, 회귀는 수치를 예측하는 문제 ex): 사진속 인물의 몸무게 예측

분류는 주로 소프트맥스 함수를 씀. -> 항상 모든 출력의 합이 1이됨: 확률로 생ㅇ각 가능!

yK - exp(ak)/sigma_i=1 to n exp(ai) n은 출력층의 뉴런수, k는 n번째 출력임을 의미 


출력층의 뉴런수: 분류하고 싶은 클래스 수로 정함!!
'''


def stepfunction(x):
    y = x>0
    return y.astype(np.int)
def sigmoid(x):
    return 1/ (1+ np.exp(-x))
def relu(x):
    return np.maximum(0,x)
def softma(a):
    c = np.max(a) # 오버플로 대책
    exp_a = np.exp(a-c)
    sum_exp = np.sum(exp_a)
    y = exp_a/ sum_exp
    return y
