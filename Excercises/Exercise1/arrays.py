# This program illustrates some common array operations in Python
import numpy as np
a0 = np.ones((3,2),dtype=int)
a1 = np.arange(6).reshape(3,2)
a2 = np.arange(8).reshape(2,4)
a3 = np.array([2,4,6]).reshape(3,1)
a4 = np.array([2,5])

print('a0')
print(a0)

print('a1')
print(a1)

print('a2')
print(a2)

print('a3')
print(a3)

print('a4')
print(a4)

# Array plus scalar
print('a1 + 5')
print(a1+5)

# Array times scalar
print('a1 * 2')
print(a1*2)

# Array plus array of equal size
print('a0 + a1')
print(a0+a1)

# Array times array of equal size
print('a0 * a1')
print(a0*a1)

# Matrix multiplication - first array is m-by-n, second is n-by-k, result is m-by-k
print('matmul(a1,a2)')
print(np.matmul(a1,a2))

# Broadcasting - one array is m-by-n, the other is n, 1-by-n or m-by-1 - result is m-by-n
print('a1 * a3')
print(a1*a3)

print('a1 * a4')
print(a1*a4)

# Boolean indexing
a5 = np.arange(6).reshape(2,3)
print('a5')
print(a5)
print('a5>2')
print(a5>2)
print('a5%2==1')
print(a5%2==1)
print('a5[a5>2]')
print(a5[a5>2])
a5[a5%2==0] = 10
print('a5')
print(a5)

