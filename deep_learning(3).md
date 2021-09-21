# 3 신경망
- - - -
퍼셉트론은 복잡한 함수도 표현 가능하지만 가중치를 설정 하는 것은 수동으로 해야함
신경망은 이 가중치의 값을 자동으로 학습

*3.1* 신경망의 예
맨 왼쪽에 입력층 가운데 층이 은닉층 그리고 오른쪽이 출력층
![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)

편향을 명시한 퍼셉트론
일반적인 퍼셉트론에 편향을 입력층에 추가
![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)

*3.2* 활성화 함수
       * 입력 신호의 총합이 활성화를 일으키는지 정하는 역할
$$ a = b + w1x1 + w2x2$$
$$ y = h(a) $$
	* 임계값을 경계로 출력이 바뀌는 계단 함수

*3.3* 시그모이드 함수
	* 입력값을 특정 값으로 바꾸는 변환기 역할
$$ h(x) = 1/ (1 +exp(-x)) $$ 
exp(-x)는 $$e^-x$$를 의미
h(1.0) == 0.731...
	* 시그모이드 함수를 이용하여 활성화함수의 신호를 변환후 다음 뉴런에 전달
	* 계단 함수 구현
```python
def step_function(x): # x의 값을 실수만 되고 넘파이 배열은 안됨
	  if x > 0:
	  else:
		  return 0
def step_function(x): # 넘파이 배열(np.array([a, b]))도 지원 가능
    y = x > 0
	  return y.astype(np.int)
```
	
	* 넘파이 배열의 부둥호 계산
```python
>>> import numpy as np
>>> x = np.array([-1.0, 1.0, 2.0])
>>> x
array([-1., 1., 2.])
>>> y = x > 0
>>> y
array([False, True, True], dtype=bool) #넘파이 계산시 bool배열 생성

>>> y = y.astype(np.int)
>>> y
array([0, 1, 1])
```

	* 계단 함수 그래프
```python
import numpy as np
import matplotlib.pylab as plt

def step_function(x):
	  return np.array(x > 0, dtype=np.int)

x = np.arange(-5.0, 5.0, 0.1) # -5부터 5까지 0.1간격으로 배열 생성
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) #y축 범위 지정
plt.show()
```

![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/image.png)


	* 시그모이드 함수 구현
```python
import numpy as np
import matplotlib.pylab as plt
def sigmold(x): #시그모이드 함수
    return 1 / (1 + np.exp(-x))


>>> x = np.array([-1.0, 1.0, 2.0])
>>> sigmold(x)
array([0.26894142], [0.73105858], [0.88079708]) # 브로드캐스트
```
	* 시그모이드 함수 구현 코드
```python
x = np.arange(-5.0, 5.0, 0.1)
y = sigmold(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1) # y축 볌위 지정
plt.show()
```
![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/image.png)

* 시그모이드 함수와 계단 함수 비교차이점
차이점
시그모이드는 연속적인 실수의 흐름
계단함수는 0,1의 값

공통점
두 함수 모두 비슷한 모양
출력값이 0과 1사이

* 비선형 함수
$$f(x)= ax + b$$로 나타낼수 없는 함수 (a, b는 상수)
시그모이드와 계단 함수는 비선형 함수
신경망에서는 비선형 함수를 사용


* ReLU 함수
최근에는 시그모이드 함수 대신 ReLU함수를 사용
입력이 0이 넘으면 입력값 그대로 출력하고 0이하면 0출력
즉 h(x) = x (x > 0) , 0 (x <= 0)

```python
def relu(x):
	  return np.maximum(0, x) # maximum함수는 두 입력중 큰값을 반환
```

*3.3* 다차원 배열의 계산

3.3.1 다차원 배열
	* 1차원 배열
```python
>>> import numpy as np
>>> A = np.array([1, 2, 3, 4])
>>> print(A)
[1 2 3 4]
>>> np.ndim(A) # 배열의 차원수 확인
1
>>> A.shape # 배열의 형상, 튜플로 반환
(4,)
>>> A.shape[0]
4
```
	* 2x2 배열을 행렬, 배열의 가로방향을 **행** 세로방향을 **열**

3.3.2 행렬의 곱
	* A = 1 2 B = 5 6 이라 하면
		3 4 	      7 8
	1*5 + 2*7 == 19
	A + B = 19 22
		       43 50
	
	* 파이썬으로 구현
```python
>>> A = np.array([[1,2],[3,4]])
>>> A.shape
(2, 2)
>>> B = np.array([[5,6],[7,8]])
>>> B.shape
(2, 3)
>>> np.dot(A, B) #행렬의 곱 계산
array([[19, 22],
		 [43, 50]])
```
	행렬 계산시 2x3이랑 3x2와는 다름
	향렬 계산시 A행렬의 열 수와 B행렬의 행수가 같아야됨
```python
>>> A = np.array([[1,2][3,4][5,6]])
>>> A.shape
(3, 2)
>>> B = np.array([7,8]) 
>>> B.shape
(2,)
>>> np.dot(A, B)
array([23, 53, 83)
```

3.3.3 신경망에서의 행렬 곱
![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)
dot(2,2x3)
```python
>>> X = np.array([1, 2])
>>> X.shape
(2,)
>>> W = np.array([[1, 3, 5],[2, 4, 6]])
>>> print(W)
[[1 3 5]
 [2 4 6]]
>>> W.shape
>>> (2, 3)
>>> Y = np.dot(X, W)
>>> print(Y)
[5 11 17]
```

**3.4** 3층 신경망 구현하기
3.4.1 표기법
![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)

3.4.2 각 신호의 신호 전달 구현하기
![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)
	* 편향의 경우 오른쪽 아래 인덱스가 다음층 뉴런밖에 없음

![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)
	* 0층에서 1층 구현
```python
X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) #(2, 3)
print(X.shape) #(2,)
print(B.shape) #(3,)

A1 = np.dot(X, W1) + B1

Z1 = sigmoid(A1)
print(A1)	#[0.3, 0.7, 1.1]
print(Z1) #[0.5744, 0.6681, 0.7502]
```
	* 1층에서 2층 구현
```python
W2 = np.array([[0.1, 0.4][0.2, 0.5][0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) #(3,)
print(W2.shape) #(3,2)
print(B2.shape) #(2,)

A2 = np.dot(Z1, W2) + B2

Z2 = sigmoid(A2)
```
	1층과 비슷

	* 2층에서 출력층 구현
```python
def identity_function(x):
    return x

W3 = np.array([0.1, 0.3], [0.2, 0.4])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3) #혹은  Y = A3
```
	활성화 함수만 다르게 설정

3.4.3 구현 정리
```python
import numpy as np

def identity_function(x):
	return x

def sigmoid(x): #시그모이드 함수
    return 1 / (1 + np.exp(-x))
    
def init_network():
	network = {} #딕셔너리 변수
	network['W1'] = np.array([[0.1, 0.3, 0.5],[0.2, 0.4, 0.6]])
	network['b1'] = np.array([0.1, 0.2, 0.3])
	network['W2'] = np.array([[0.1, 0.4],[0.2, 0.5], [0.3, 0.6]])
	network['b2'] = np.array([0.1, 0.2])
	network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
	network['b3'] = np.array([0.1, 0.2])
	
	return network
	
def forward(network, x):
	W1, W2, W3 = network['W1'], network['W2'], network['W3']
	b1, b2, b3 = network['b1'], network['b2'], network['b3']
	
	a1 = np.dot(x, W1) + b1
	z1 = sigmoid(a1)
	a2 = np.dot(z1, W2) + b2
	z2 = sigmoid(a2)
	a3 = np.dot(z2, W3) + b3
	y = identity_function(a3)
	
	return y
	
network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y) #[ 0.31682708  0.69627909]
```

**3.5** 출력층 계산
	신경망은 분류, 회귀 모두 사용가능
	분류는 출력층의 함수가 소프트맥스 함수, 회귀는 항등 함수

3.5.1 항등 함수와 소프트맥스 함수 구현
	항등함수는 입력과 출력이 같은 y=x함수
	소프트맥스 함수
	![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)

```python
>>> a = np.array([0.3, 2.9, 4.0])
>>>
>>> exp_a = np.exp(a)
>>> print(exp_a)
[1.3498 18.1741 54.598]
>>>
>>> sum_exp_a = np.sum(exp_a)
74.1221
>>>
>>> y = exp_a / sum_exp_a
>>> print(y)
[0.018 0.2451 0.7365]
```

	함수로 구현
```python
def softmax(a):
exp_a = np.exp(a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
```

3.5.2 소프트맥스 함수 구현시 주의점
	* 오버플로 가능성 높음
![](*3*%20%EC%8B%A0%EA%B2%BD%EB%A7%9D/bear_sketch@2x.png)
	* 지수 함수를 계산할때 특정 정수를 더해도 변화 없음
	* 일반적으로 입력 신호 중 최대값을 사용
```python
>>> a = np.array([1010, 1000, 900])
>>> c = np.max(a)
>>> a - c
array([0, -10, -20])
>>>np.exp(a - c) / np.sum(np.exp(a - c))
array([9.9995, 4.5397, 2.0610])
```
다시 구현

```python
def softmax(a):
c = np.max(a)
exp_a = np.exp(a - c)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
```
