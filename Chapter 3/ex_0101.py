import numpy as np
np.__version__

#1.1 NumPPy basics:

# [] là list, () là tuple
# kết quả giống nhau đều tạo numpy array
arr1 = np.array([1,3,5,7,9]) #PHổ biến hơn
arr2 = np.array((1,3,5,7,9))
type(arr1)
arr3 = arr1
id(arr1)
id(arr3)
arr4 = arr1.copy()
id(arr4)

#1.2 Creating NumPy arrays:
#numpy.arange() function:

np.arange(10)
np.arange(10, 20)
np.arange(10, 20, 3)
arr = np.arange(10)
len(arr)
arr.size
arr

#numpy.linspace() function
np.linspace(1, 10, 5) # từ 1 đến 10 tạo 5 số cách đều nhau
# array([ 1, . 3.25, 5.5 , 7.75, 10. ])
arr = np.linspace(0, 10, 5, retstep=True)
# retstep = False thì hàm trả về mảng numpy
# retstep = True thì hàm trả về một tuple(chỉ có 2 phần tử)
type(arr)
arr[0]
arr[1]

#numpy.zeros() and numpy.ones() functions:
np.zeros(10)
np.zeros((3,4))
np.zeros(5, dtype ='int64')
np.zeros(5, dtype ='int64').dtype
arr = np.ones((2,3), dtype ='int_')
arr
arr.dtype
arr.astype('float32')

#Data type
arr1 = np.array([111, 2.3,  True, False, False])
arr1
arr2 = np.array([111, 2.3, 'python', 'abc'])
arr2arr3 = np.array([111, True, 'abc'])
arr3

#1.4. Indexing and slicing NumPy arrays
a = np.array([1, 2, 3, 4, 5])
a[2]
a[-1]
a[:]
a = np.array([[1, 2, 3,], [4, 5, 6], [7, 8, 9]])
a
a[1]
a[-1]
a[:2]
a[2][1]
a[2,1]
a[[0,2]] # hàng 0 và 2
a[1:,1:]
#Fancy indexing
arr = np.arange(100)
arrMask = ( (arr % 5) == 0 )
arr[arrMask]
arrMask = ( ( (arr % 5) == 0 )  & ( arr >0) )
arr[arrMask]

#1.4. NumPy attributes and reshaping
a.size
a.shape
a.ndim

a = np.arange(15)
a.reshape(3,5)

a
a = a.reshape(3,5)
a

a = np.array([2,5,1,3])
a.shape

a = a.reshape((2,2)) #khác về cách thức truyền đôi một so với a.reshape(2,2)

a = np.arange(10)
b = a.reshape(2,5)
a[0] = -999
b
# Make a copy in order to create a new object.
c = a.reshape(2,5).copy()
c[0,0] = 0
a

#1.5. Modifying NumPy arrays
a = np.array([1, 2, 3])
b = np.append(a, [4, 5, 6])
b
a = np.array([[1, 2], [3, 4]])
b = np.append(a, [[9, 9]], axis=0)
b
c = np.append(a, [[9], [9]] , axis=1)
c

a = np.array([[1, 2, 3],[4, 5, 6]])
np.delete(a, 0)
np.delete(a, (0, 2, 4))
np.delete(a, 0, axis = 0)
np.delete(a, 1, axis = 1)