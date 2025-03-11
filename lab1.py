from time import time
import numpy as np

v = np.float32(0.31244124124121241242)
#v = np.float32(0.53125)
N = 10**7
T = [v for _ in range(N)]

def khan(N):
  sum = np.float32(0.0)
  err = np.float32(0.0)
  for i in range(N):
    y = np.float32(T[i] - err)
    temp = np.float32(sum + y)
    err = np.float32((temp - sum) - y)
    sum = temp
  return sum

start = time()
res = khan(N)

print("B" ,(abs(res - v*N)))
print("W", abs(res - v*N)/(v*10**7)*100 ,"%")
print(time()-start," s")