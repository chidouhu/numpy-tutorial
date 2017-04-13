# Numpy Quickstart Tutorial

## 基础

NumPy的主要目标是同构的多维数组. 它是多个元素(通常是数字)的一个表格, 所有的元素都是相同类型, 并由一个非负整数的tuple来索引的. 在NumPy中维度被称为**axes**, axes的个数称为**rank**.

举个例子, 3D场景中的一个点坐标[1, 2, 1]是一个rank为1的数组, 因为它只有一个axis. 该axis的长度为3, 在下面的例子中, 数组的rank为2(这是2维). 第一维(axis)的长度为2, 第二维的长度为3 (TODO. ???).

```
[[1., 0., 0.],
 [0., 1., 2.]]
```

NumPy的数组类称为**ndarray**. 通常它也用来作为**array**的别名. 注意**numpy.array**和标准Python库的类**array.array**不同, 后者只处理一维数组, 功能也更少. **ndarray**对象的更多属性如下:

1. Ndarray.ndim
   1. array的axes(维度)个数. 在Python世界中, 维度的个数也称为**rank**.
2. Ndarray.shape
   1. array的维数. 通常是一个整形的tuple来描述array在每一维的大小. 对于一个n行m列的矩阵来说, **shape**就是**(n, m)**. **shape**元组的长度也就是**rank**, 或者说是维度的个数, **ndim**.
3. Ndarray.size
   1. array元素的总个数. 它等于**shape**元素的乘积.
4. Ndarray.dtype
   1. 一个用于描述array元素类型的对象. 你可以用标准的Python类型来创建或者描述一个dtype. 此外Numpy还提供自己的类型. Numpy.int32, numpy.int16及numpy.float64等等.
5. Ndarray.itemsize
   1. array中每个元素的字节数. 例如, 由**float64**类型组成的array的**itemsize**就是8 (=64/8), **complex32**类型的**itemsize**就是4 (=32/8). 它和**ndarray.dtype.itemsize**相等.
6. Ndarray.data
   1. 包含array实际元素的缓冲区. 通常, 我们不需要使用这个属性, 因为我们只需要直接通过索引来获取元素就可以了.

### 一个例子

~~~python
>>> import numpy as np
>>> a = np.arange(15).reshape(3, 5)
>>> a
array([[0, 1, 2, 3, 4],
       [5, 6, 7, 8, 9],
       [10, 11, 12, 13, 14]])
>>> a.shape
(3, 5)
>>> a.ndim
2
>>> a.dtype.name
'int64'
>>> a.itemsize
8
>>> a.size
15
>>> type(a)
<type 'numpy.ndarray'>
>>> b = np.array([6, 7, 8])
b
array([6, 7, 8])
>>> type(b)
<type 'numpy.ndarray'>
~~~

### Array的创建

有几种创建array的方法.

例如, 你可以对一个标准的Python list或者tuple使用array方法来创建array. 生成array的类型从元素序列的类型中获取.

~~~python
>>> import numpy as np
>>> a = np.array([2, 3, 4])
>>> a
array([2, 3, 4])
>>> a.dtype
dtype('int64')
>>> b = np.array([1.2, 3.5, 5.1])
>>> b.dtype
dtype('float64')
~~~

一个常见的错误是用多个数字参数来生成array, 而不是用一个数字的list作为参数.

~~~Python
>>> a = np.array(1, 2, 3, 4)	# WRONG
>>> a = np.array([1, 2, 3, 4])	# RIGHT
~~~

**array**将序列的序列转化为二维数组, 如果是序列的序列的序列就转化为三维数组, 以此类推.

~~~Python
>>> b = np.array([(1.5, 2, 3), (4, 5, 6)])
>>> b
array([[ 1.5, 2. , 3. ],
       [ 4. , 5. , 6. ]])
~~~

array的类型也可以在创建时显示指定:

~~~Python
>>> c = np.array([[1,2], [3,4]], dtype=complex)
>>> c
array([[ 1.+0.j, 2.+0.j],
       [ 3.+0.j, 4.+0.j]])
~~~

通常, array的元素在一上来是未知的, 但是大小是已知的. 因此, NumPy提供了多种方法来用初始占位指创建array. 这样最小化了扩张array的需要, 而该操作的代价是很高的.

**zeros**方法创建一个全由0组成的array, 而**ones**方法创建一个全由1组成的array, 而**empty**方法创建一个初始值随机(依赖内存状态)的array. 默认情况下, dtype为**float64**.

~~~python
>>> np.zeros( (3,4) )
array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]])
>>> np.ones( (2,3,4), dtype=np.int16)	# 同时指定了类型
array([[[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]],
       [[ 1, 1, 1, 1],
        [ 1, 1, 1, 1],
        [ 1, 1, 1, 1]]], dtype=int16)
>>> np.empty( (2,3) )                   # uninitialized, output may vary
array([[  3.73603959e-262,   6.02658058e-154,   6.55490914e-260],
       [  5.30498948e-313,   3.14673309e-307,   1.00000000e+000]])
~~~

NumPy还提供了一个类**range**方法来创建数字序列.

~~~
>>> np.arange( 10, 30, 5 )
array([10, 15, 20, 25])
>>> np.arange( 0, 2, 0.3 )                 # it accepts float arguments
array([ 0. ,  0.3,  0.6,  0.9,  1.2,  1.5,  1.8])
~~~

当**arange**使用浮点数作为参数时, 由于浮点数精度的原因, 通常无法精确预测包含的元素个数. 因此, 最好使用方法**linspace**

### 基本操作

对array的数值操作会作用到每个元素上. 结果会创建一个新的array并用结果填满.

~~~python
>>> a = np.array([20, 30, 40, 50])
>>> b = np.arange(4)
>>> b
array([0, 1, 2, 3])
>>> c = a - b
>>> c
array([20, 29, 38, 47])
>>> b ** 2
array([0, 1, 4, 9])
>>> 10 * np.sin(a)
array([ 9.12945251, -9.88031624,  7.4511316 , -2.62374854])
>>> a<35
array([ True, True, False, False], dtype=bool)
~~~

和很多矩阵语言不同, 叉积操作*****是对NumPy array的每个元素操作. 真正的矩阵乘可以通过**dot**方法:

~~~python
>>> A = np.array([[1, 1],
                [0, 1]])
>>> B = np.array( [[2,0],
...             [3,4]] )
>>> A*B                         # elementwise product
array([[2, 0],
       [0, 4]])
>>> A.dot(B)                    # matrix product
array([[5, 4],
       [3, 4]])
>>> np.dot(A, B)                # another matrix product
array([[5, 4],
       [3, 4]])
~~~

某些操作, 如**+=**和***=**, 是对当前已有的array操作而不是创建一个新的.

~~~python
>>> a = np.ones((2,3), dtype=int)
>>> b = np.random.random((2,3))
>>> a *= 3
>>> a
array([[3, 3, 3],
       [3, 3, 3]])
>>> b += a
>>> b
array([[ 3.417022  ,  3.72032449,  3.00011437],
       [ 3.30233257,  3.14675589,  3.09233859]])
>>> a += b                  # b is not automatically converted to integer type
Traceback (most recent call last):
  ...
TypeError: Cannot cast ufunc add output from dtype('float64') to dtype('int64') with casting rule 'same_kind'
~~~

当对不同类型的array进行操作时, 生成的array类型通常是和更精确的类型相关(类似于upcasting).

~~~python
>>> a = np.ones(3, dtype=np.int32)
>>> b = np.linspace(0,pi,3)
>>> b.dtype.name
'float64'
>>> c = a+b
>>> c
array([ 1.        ,  2.57079633,  4.14159265])
>>> c.dtype.name
'float64'
>>> d = np.exp(c*1j)
>>> d
array([ 0.54030231+0.84147098j, -0.84147098+0.54030231j,
       -0.54030231-0.84147098j])
>>> d.dtype.name
'complex128'
~~~

许多单运算符操作, 如计算array每个元素的和, 也都在**ndarray**类中实现了.

~~~Python
>>> a = np.random.random((2,3))
>>> a
array([[ 0.18626021,  0.34556073,  0.39676747],
       [ 0.53881673,  0.41919451,  0.6852195 ]])
>>> a.sum()
2.5718191614547998
>>> a.min()
0.1862602113776709
>>> a.max()
0.6852195003967595
~~~

默认情况下, 无论array的shape是什么样的, 这些操作会应用到整个array上, 尽管它是一个数字的列表. 然而, 你可以通过制定**axis**参数来对array的特定axis操作.

~~~python
>>> b = np.arange(12).reshape(3,4)
>>> b
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>>
>>> b.sum(axis=0)                            # sum of each column
array([12, 15, 18, 21])
>>>
>>> b.min(axis=1)                            # min of each row
array([0, 4, 8])
>>>
>>> b.cumsum(axis=1)                         # cumulative sum along each row
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38]])
~~~

### Universal方法

NumPy提供了大家熟悉的数学方法如sin, cos和exp. 在NumPy中, 这类操作称之为"Universal方法"(**ufunc**). 通过NumPy, 这些方法会作用到array的每个元素上, 生成一个array作为输出.

~~~python
>>> B = np.arange(3)
>>> B
array([0, 1, 2])
>>> np.exp(B)
array([ 1.        ,  2.71828183,  7.3890561 ])
>>> np.sqrt(B)
array([ 0.        ,  1.        ,  1.41421356])
>>> C = np.array([2., -1., 4.])
>>> np.add(B, C)
array([ 2.,  0.,  6.])
~~~

### 索引, 切片和迭代

**一维**array可以被索引, 切片和迭代, 类似于list和其他Python顺序结构.

~~~python
>>> a = np.arange(10)**3
>>> a
array([  0,   1,   8,  27,  64, 125, 216, 343, 512, 729])
>>> a[2]
8
>>> a[2:5]
array([ 8, 27, 64])
>>> a[:6:2] = -1000    # equivalent to a[0:6:2] = -1000; from start to position 6, exclusive, set every 2nd element to -1000
>>> a
array([-1000,     1, -1000,    27, -1000,   125,   216,   343,   512,   729])
>>> a[ : :-1]                                 # reversed a
array([  729,   512,   343,   216,   125, -1000,    27, -1000,     1, -1000])
>>> for i in a:
...     print(i**(1/3.))
...
nan
1.0
nan
3.0
nan
5.0
6.0
7.0
8.0
9.0
~~~

**多维**array在每个axis上都有一个索引. 这些下标是一个由逗号分隔的tuple.

~~~python
>>> def f(x,y):
...     return 10*x+y
...
>>> b = np.fromfunction(f,(5,4),dtype=int)
>>> b
array([[ 0,  1,  2,  3],
       [10, 11, 12, 13],
       [20, 21, 22, 23],
       [30, 31, 32, 33],
       [40, 41, 42, 43]])
>>> b[2,3]
23
>>> b[0:5, 1]                       # each row in the second column of b
array([ 1, 11, 21, 31, 41])
>>> b[ : ,1]                        # equivalent to the previous example
array([ 1, 11, 21, 31, 41])
>>> b[1:3, : ]                      # each column in the second and third row of b
array([[10, 11, 12, 13],
       [20, 21, 22, 23]])
~~~

如果没有提供axes个数, 缺省的下标意味着整个slice:

~~~python
>>> b[-1]                                  # the last row. Equivalent to b[-1,:]
array([40, 41, 42, 43])
~~~

在**b[i]**这种通过括号的描述中, **i**后面跟着很多**:**实例来描述余下的axes. NumPy也允许你通过连续点的方式来描述, 比如**b[i, …]**.

连续点(**…**)表示构成完整索引元组的剩余元素. 比如, 如果**x**是一个rank为5的array(例如, 它有5个axes), 那么

* **x[1,2,…]**等价于**x[1,2,:,:,:,:]**,
* **x[…,3]**等价于**x[:,:,:,:,3]**,
* **x[4,…,5,:]**等价于**x[4,:,:,5,:]**.

~~~python
>>> c = np.array( [[[  0,  1,  2],               # a 3D array (two stacked 2D arrays)
...                 [ 10, 12, 13]],
...                [[100,101,102],
...                 [110,112,113]]])
>>> c.shape
(2, 2, 3)
>>> c[1,...]                                   # same as c[1,:,:] or c[1]
array([[100, 101, 102],
       [110, 112, 113]])
>>> c[...,2]                                   # same as c[:,:,2]
array([[  2,  13],
       [102, 113]])
~~~

对多维array的**迭代**是通过第一个axis完成的:

~~~python
>>> for row in b:
...     print(row)
...
[0 1 2 3]
[10 11 12 13]
[20 21 22 23]
[30 31 32 33]
[40 41 42 43]
~~~

然而,  如果你希望对array中的每个元素做一个操作, 你可以通过**flat**属性来对array的每个元素做一次迭代:

~~~python
>>> for element in b.flat:
...     print(element)
...
0
1
2
3
10
11
12
13
20
21
22
23
30
31
32
33
40
41
42
43
~~~

(TODO. Indexing)

## Shape操作

### 改变array的shape

array的shape是通过每个axis的元素个数得到的:

~~~python
>>> a = np.floor(10*np.random.random((3,4)))
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.shape
(3, 4)
~~~

array的shape可以通过某些命令来改变. 注意下面三个命令都会返回一个修改的array, 不会修改原有的array:

~~~python
>>> a.ravel()  # returns the array, flattened
array([ 2.,  8.,  0.,  6.,  4.,  5.,  1.,  1.,  8.,  9.,  3.,  6.])
>>> a.reshape(6,2)  # returns the array with a modified shape
array([[ 2.,  8.],
       [ 0.,  6.],
       [ 4.,  5.],
       [ 1.,  1.],
       [ 8.,  9.],
       [ 3.,  6.]])
>>> a.T  # returns the array, transposed
array([[ 2.,  4.,  8.],
       [ 8.,  5.,  9.],
       [ 0.,  1.,  3.],
       [ 6.,  1.,  6.]])
>>> a.T.shape
(4, 3)
>>> a.shape
(3, 4)
~~~

revel()结果的元素顺序是类似于"C风格"的, 也就是最右侧的索引"变化最快", 因此a[0,0]后面的元素是a[0,1]. 如果array被reshape为另一种shape, array还是被当做"C风格"来对待. NumPy通常会以这种顺序来存储array, 因此revel()通常不需要拷贝元素, 但是如果array是来自于另一个array的片段, 或者是以特殊的方式创建的, 它可能需要拷贝. revel和reshape方法也可以通过一个可选参数来使用"FORTRAN风格"的array, 这样是最左侧的索引"变化最快".

reshape方法会返回一个新的shape, 而ndarray.resize方法会修改array本身:

~~~python
>>> a
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
>>> a.resize((2,6))
>>> a
array([[ 2.,  8.,  0.,  6.,  4.,  5.],
       [ 1.,  1.,  8.,  9.,  3.,  6.]])
~~~

如果在reshape操作中维度参数给了-1, 会自动计算其他维度:

~~~python
>>> a.reshape(3,-1)
array([[ 2.,  8.,  0.,  6.],
       [ 4.,  5.,  1.,  1.],
       [ 8.,  9.,  3.,  6.]])
~~~

### 合并多个array

多个array可以沿着不同的axes合并:

~~~python
>>> a = np.floor(10*np.random.random((2,2)))
>>> a
array([[ 8.,  8.],
       [ 0.,  0.]])
>>> b = np.floor(10*np.random.random((2,2)))
>>> b
array([[ 1.,  8.],
       [ 0.,  4.]])
>>> np.vstack((a,b))
array([[ 8.,  8.],
       [ 0.,  0.],
       [ 1.,  8.],
       [ 0.,  4.]])
>>> np.hstack((a,b))
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
~~~

函数column_stack将一维数组插入到一个二维数组中作为一列. 这和一维数组的vstack作用相同:

~~~python
>>> from numpy import newaxis
>>> np.column_stack((a,b))   # With 2D arrays
array([[ 8.,  8.,  1.,  8.],
       [ 0.,  0.,  0.,  4.]])
>>> a = np.array([4.,2.])
>>> b = np.array([2.,8.])
>>> a[:,newaxis]  # This allows to have a 2D columns vector
array([[ 4.],
       [ 2.]])
>>> np.column_stack((a[:,newaxis],b[:,newaxis]))
array([[ 4.,  2.],
       [ 2.,  8.]])
>>> np.vstack((a[:,newaxis],b[:,newaxis])) # The behavior of vstack is different
array([[ 4.],
       [ 2.],
       [ 2.],
       [ 8.]])
~~~

对于维度超过2的array来说, hstack沿着第二个axes插入, vstack沿着第一维插入, concatenate允许通过可选参数来指定沿着哪一个axis来插入.

#### 注意

在复杂场景下, r_ 和 c_ 适用于来沿着某一个axis插入. 它们允许使用range语法(":").

~~~python
>>> np.r_[1:4,0,4]
array([1, 2, 3, 0, 4])
~~~

当使用array作为参数时, r_ 和 c_类似于vstack和hstack的默认行为, 但是允许可选参数来指定.

### 将一个array切分成多个小的array

使用hsplit, 你可以沿水平轴来切分数组, 可以通过制定返回的array尺寸来切分, 也可以指定在某一列后切分:

~~~python
>>> a = np.floor(10*np.random.random((2,12)))
>>> a
array([[ 9.,  5.,  6.,  3.,  6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 1.,  4.,  9.,  2.,  2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])
>>> np.hsplit(a,3)   # Split a into 3
[array([[ 9.,  5.,  6.,  3.],
       [ 1.,  4.,  9.,  2.]]), array([[ 6.,  8.,  0.,  7.],
       [ 2.,  1.,  0.,  6.]]), array([[ 9.,  7.,  2.,  7.],
       [ 2.,  2.,  4.,  0.]])]
>>> np.hsplit(a,(3,4))   # Split a after the third and the fourth column
[array([[ 9.,  5.,  6.],
       [ 1.,  4.,  9.]]), array([[ 3.],
       [ 2.]]), array([[ 6.,  8.,  0.,  7.,  9.,  7.,  2.,  7.],
       [ 2.,  1.,  0.,  6.,  2.,  2.,  4.,  0.]])]
~~~

vsplit沿着纵轴来切分axis, 而array_split允许你沿着某一个axis来切分.

## 拷贝与视图

在操作和生成array时, 它们的data有时会被拷贝到新array, 但有时不会. 这经常会让初学者混淆. 有三种情况:

### 完全不拷贝

简单的赋值操作不会拷贝array对象.

~~~python
>>> a = np.arange(12)
>>> b = a            # no new object is created
>>> b is a           # a and b are two names for the same ndarray object
True
>>> b.shape = 3,4    # changes the shape of a
>>> a.shape
(3, 4)
~~~

Python在函数调用的时候将可变对象以引用的方式传递, 因此函数调用也没有拷贝.

~~~python
>>> def f(x):
...     print(id(x))
...
>>> id(a)                           # id is a unique identifier of an object
148293216
>>> f(a)
148293216
~~~

### 视图或Shallow拷贝

不同的array对象可以共享同一份数据. **view**方法创建一个新的array对象并管理相同的data.

~~~python
>>> c = a.view()
>>> c is a
False
>>> c.base is a                        # c is a view of the data owned by a
True
>>> c.flags.owndata
False
>>>
>>> c.shape = 2,6                      # a's shape doesn't change
>>> a.shape
(3, 4)
>>> c[0,4] = 1234                      # a's data changes
>>> a
array([[   0,    1,    2,    3],
       [1234,    5,    6,    7],
       [   8,    9,   10,   11]])
~~~

对array的切分返回一个view:

~~~python
>>> s = a[ : , 1:3]     # spaces added for clarity; could also be written "s = a[:,1:3]"
>>> s[:] = 10           # s[:] is a view of s. Note the difference between s=10 and s[:]=10
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
~~~

### 深拷贝

**copy**方法对array和数据做整体的拷贝.

~~~python
>>> d = a.copy()                          # a new array object with new data is created
>>> d is a
False
>>> d.base is a                           # d doesn't share anything with a
False
>>> d[0,0] = 9999
>>> a
array([[   0,   10,   10,    3],
       [1234,   10,   10,    7],
       [   8,   10,   10,   11]])
~~~

### 函数与方法回顾

(TODO)

## Less Basic

### 广播规则

广播允许用universal方法来处理

(TODO)

## Fancy索引和索引trick

NumPy提供了比标准Python顺序结构更灵活的索引方法. 除了整形和切片索引, 如我们前面所述, array也可以用整形array或者bool array来索引.

### 通过Array的下标做索引

~~~python
>>> a = np.arange(12)**2                       # the first 12 square numbers
>>> i = np.array( [ 1,1,3,8,5 ] )              # an array of indices
>>> a[i]                                       # the elements of a at the positions i
array([ 1,  1,  9, 64, 25])
>>>
>>> j = np.array( [ [ 3, 4], [ 9, 7 ] ] )      # a bidimensional array of indices
>>> a[j]                                       # the same shape as j
array([[ 9, 16],
       [81, 49]])
~~~

如果被索引的数组**a**是多维的, 

### 通过Boolean数组做索引

### ix_()方法

### 通过strings索引

## 线性代数

### 简单的数组操作

## 技巧

### "自动"reshape

### 向量合并

### Histograms

### 未来阅读









