from time import time
import numpy as np

S = [2,3.6667,5,7.2,10]
N = [50,100,200,500,1000]

def f32(s,n):
    sum1 = np.float32(0)
    for i in range(1,n+1):
        sum1 += np.float32(((-1)**i) * (1/np.float32(i ** s)))
    sum2 = np.float32(0)
    for i in range(1,n+1):
        sum2 += np.float32(1/np.float32(i ** s))
# return eta/zeta
    return sum1,sum2

def f64(s,n):
    sum1 = np.float64(0)
    for i in range(1,n+1):
        sum1 += np.float64(((-1)**i) * (1/np.float64(i ** s)))
    sum2 = np.float64(0)
    for i in range(1,n+1):
        sum2 += np.float64(1/np.float64(i ** s))
# return eta/zeta
    return sum1,sum2

def f64R(s,n):
    sum1 = np.float64(0)
    for i in range(n+1,0,-1):
        sum1 += np.float64(((-1)**i) * (1/np.float64(i ** s)))
    sum2 = np.float64(0)
    for i in range(n+1,0,-1):
        sum2 += np.float64(1/np.float64(i ** s))
# return eta/zeta
    return sum1,sum2


def f32R(s,n):
    sum1 = np.float32(0)
    for i in range(n+1,0,-1):
        sum1 += np.float32(((-1)**i) * (1/np.float32(i ** s)))
    sum2 = np.float32(0)
    for i in range(n+1,0,-1):
        sum2 += np.float32(1/np.float32(i ** s))
# return eta/zeta
    return sum1,sum2

for s in S:
    for n in N:
        print("f32, eta, zeta",f32(s,n))
        print("f64, eta, zeta",f64(s,n))
        print("f32 reversed, eta, zeta",f32R(s,n))
        print("f64 reversed, eta, zeta",f64R(s,n))
    print("\n\n")