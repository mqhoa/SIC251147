#1. NumPy array operatioins
#1.1. '+' and '*' operators for Python lists
import numpy as np
a=[1,2,3]
a+a
a*3

b=np.array([1,2,3])
b*3
b+b
np.repeat(b,3)

#1.2. '+' and '*' operators for NumPy arrays:
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
a+b
a-b
a*b
1.0*a/b

#1.3. Universal functions:
x = np.array([0, 1, 2, 3, 4, 5])
np.sqrt(x)
np.exp(x) #e^x
type(np.sqrt)
type(np.exp)

#1.4. NumPy functions:
np.random.seed(123)#có thể thay 123, dùng để cố định dãy ngẫu nhiên random bên dưới
a = np.random.randint(1, 11, size=1000) #mảng gồm 1000 số nguyên từ 1 đến 11(có thể trùng lặp)
np.unique(a)

