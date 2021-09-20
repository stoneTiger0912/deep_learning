# numpy
-------
넘파이 가져오기
```python
>>> import numpy as np #넘파이 라이브러리
```
  
넘파이 배열 생성
```python
>>> x = np.array([1.0,2.0,3.0]) #넘파이 배열을 만들경우 파이썬 리스트를 받아 np.array() 메서드를 사용
>>> print(x)
[1. 2. 3.]
>>> type(x)
<class 'numpy.ndarray'> #넘파이가 제공하는 특수 형태의 배열을 반환
```
  
넘파이 산술연산
```python
>>> x = np.array([1.0, 2.0, 3.0])
>>> y = np.array([4.0, 5.0, 6.0])
>>> x + y
array([5., 7., 9.]) #일반적인 리스트 사칙연산과 같이 원소끼리 계산
                    #사칙연산 모두 적용
                    #이런 기능을 브로드 캐스트
```
  
넘파이의 N차원 배열
```python
>>> A = np.array([[1,2], [3.4]])
>>> print(A) 
[[1 2]
 [3 4]]
>>> A.shape #A의 형태를 나타내는 함수
(2, 2)
>>> A.dtype
dtype('int64') #자료형으로 int64


>>> B = np.array([5,6], [7,8])
>>> A + B #형태가 같은 행렬일 경우 사칙연산 가능
array([[6,8],
       [10,12])
       
>>> A * 10 #행렬과 스칼라값과 연산 가능
array ([[10, 20],
        [30, 40]])
```
  
브로드캐스트: 형태가 다른 배열끼리 계산
ex) 2x2행렬 과 1x1행렬
```python
>>> A = np.array([[1,2], [3,4]])
>>> B = np.array([10,20])
>>> A * B #A의 첫번째 열이 10과 계산되고 A의 두번째 열과 20이 계산
array([10, 40,
      [30, 80]])
```
  
원소접근
```python
>>> X = np.array([[1,2],[3,4],[5,6]]) #파이썬의 리스트와 같이 원소 접근 가능
>>> X[0]
array([1,2])
>>>x[0][1]
2

>>> for row in X: #for문으로도 접근 가능
        print(row)
[1 2]
[3 4]
[5 6]
  
>>> X = X.flatten() #x를 1차원 배열로 변환
>>>print(X)
[1 2 3 4 5 6]

>>> X[np.array([0, 2, 4])] #인덱스가 0 2 4인 원소 얻기
array([1 3 5])

>>> X % 2 == 0 #특정 조건을 만족하는 원소만 얻기
array([False,  True, False,  True, False,  True])
>>> X[X % 2 == 0]
array([2,4,6])
```
  
파이썬은 C나 C++보다 속도가 느려서 C/C++로 만든 넘파이를 사용
# matplotlib
- - - -
그래프를 그려주는 라이브러리

```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1) #0부터 6까지 0.1의 단위로 생성
y = np.sin(x) #sin함수에 대입

#그래프 그리기
plt.plot(x, y)
plt.show()
```
![](matplotlib/image.png)

pyplot의 기능
```python
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label="sin")
plt.plot(x, y2, linestyle="--", label="cos") #점선으로 출력
plt.xlabel("x") #x축 이름
plt.ylabel("y") #y축 이름
plt.title('sin & cos') #제목
plt.legend() #그래프의 범례(우측 상단) 표시
plt.show()
```
![](matplotlib/image.png)
이미지 표시하기
```python
import matplotlib.pyplot as plt
from matplotlib.image import imread

img = imread('a.png') #이미지 경로

plt.imshow(img)
plt.show()
```

그외에 참조 할것들
넘파이
파이썬 라이브러리를 활용한 데이터분석
<scipy 강의 노트>


