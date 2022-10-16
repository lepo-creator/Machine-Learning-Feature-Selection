import numpy as np

a = np.arange(1, 4)
print(a)
b = np.arange(4, 6)   
print(b)                                        

c=np.insert(np.resize(a, (b.shape[0], a.shape[0])), 2, b, axis=1)
print(c)                                                                       
  