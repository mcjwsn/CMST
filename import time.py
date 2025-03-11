from time import time
import matplotlib.pyplot as plt
import numpy as np

v = np.float32(0.31244124124121241242)
#v = np.float32(0.531)
N = 10**7

start = time()
res = np.float32(0)
P = []
BZ = []
WZ = []
for i in range(N):
  if i % 25000 == 1:
    P.append(i)
    BZ.append(abs(res - v*i))
    WZ.append(abs(res - v*i)/(v*i)*100)
  res += v

#wykresy
fig, ax = plt.subplots()
ax.plot(P,WZ)

fig2, ax2 = plt.subplots()
ax2.plot(P,BZ)

plt.show()

print("Bzwgl " , abs(res - v*N))
print("Wzgl " , abs(res - v*N)/(v*N)*100 , "%")
print(time()-start, "s")


print("\n\n\n\n")

def r(T):
    if len(T) == 1: return T[-1]
    else:
        return np.float32(r(T[0:len(T)//2]) + r(T[len(T)//2:len(T)]))

T = [v] * N    
start = time()
res = r(T)
print("bezwgledny", res - N*v)
print("wzgledny", 100*abs(res - N*v)/(N*v),"%")
print(time()-start,"s")