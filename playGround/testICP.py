import numpy as np
import torch
import pytorch3d
from pytorch3d.ops import iterative_closest_point

a,b=[],[]
N=1000
for i in range(N):
   a.append([0,i,0])
   #b.append([0,i/np.sqrt(2),i/np.sqrt(2)])
   b.append([0,0,i])
a=torch.tensor(a).reshape(1,N,3).float().repeat(1,1,1)
b=torch.tensor(b).reshape(1,N,3).float().repeat(1,1,1)
icpout=iterative_closest_point(a,b)
print(icpout.RTs.T)
