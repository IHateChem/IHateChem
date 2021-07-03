import numpy as np

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


test(XOR)

    
    
    
    
    
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