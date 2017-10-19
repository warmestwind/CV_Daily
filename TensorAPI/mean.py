import numpy as np
a=[1,2,3]
b=[1,2,3]
c=np.vstack((a,b))
d=c.mean(0)
e=d.mean(0)
r=c[:,0]
print(np.shape(a))
print(r)
