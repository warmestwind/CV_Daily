import numpy as np 

trX = np.linspace(-1, 1, 101) 
num=*trX.shape  #error : cant use starred expression 
print(num)  

print(*trX.shape) #unpack arguments
