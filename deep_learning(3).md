# 3 신경망
- - - -
퍼셉트론은 복잡한 함수도 표현 가능하지만 가중치를 설정 하는 것은 수동으로 해야함
신경망은 이 가중치의 값을 자동으로 학습

*3.1* 신경망의 예
맨 왼쪽에 입력층 가운데 층이 은닉층 그리고 오른쪽이 출력층
![image](https://user-images.githubusercontent.com/91014308/134168417-f7e5a0b2-8865-48ff-afd1-c5697529646e.png)

편향을 명시한 퍼셉트론
일반적인 퍼셉트론에 편향을 입력층에 추가
![image](https://user-images.githubusercontent.com/91014308/134168522-30ee8450-8777-4de9-ac57-c7c991a985b2.png)

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
![image](https://user-images.githubusercontent.com/91014308/134168648-3e6331ab-2773-473d-af51-c543ca698f25.png)


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
![image](https://user-images.githubusercontent.com/91014308/134168710-31e2ef5c-bc24-4968-83df-28e8f8596068.png)

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
![image](https://user-images.githubusercontent.com/91014308/134168781-f960172e-49ea-4dc2-bffa-7074f37b9f92.png)
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
![image](https://user-images.githubusercontent.com/91014308/134168867-9d0fbe60-63ff-4173-8d2d-e737b2fdbee6.png)

3.4.2 각 신호의 신호 전달 구현하기
![image](https://user-images.githubusercontent.com/91014308/134168934-d8d303cb-8275-4c62-97b2-b5e8b825322c.png)
	* 편향의 경우 오른쪽 아래 인덱스가 다음층 뉴런밖에 없음

![image](https://user-images.githubusercontent.com/91014308/134168989-7bb2ba37-e3d6-456d-bb76-7066e8fd2a49.png)
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
	![image](https://user-images.githubusercontent.com/91014308/134169072-a02cf8a9-5f22-4e05-9525-41ce0480e57d.png)

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
	* 대안으로 다음과 같이 활용
![image](https://user-images.githubusercontent.com/91014308/134169147-31bd3fcc-e296-4f76-99f3-7f8b6de48d11.png)
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


소프트맥스 함수 특징
```python
>>> a = np.array([0.3, 2.9, 40])
>>> y = softmax(a)
>>> print(y)
[ 0.1821 0.2451 0.7365]
>>> np.sum(y)
1.0
```
	출력값은 0~1사이
	출력의 총합은 1
	확률로 해석 가능
	단조 증가 함수이기 때문에 대소관계는 변하지 않음

**3.6** 손글씨 숫자 인식
	* MNIST 데이터셋
	손글씨 숫자 이미지 집합
```python
import sys, os
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

def load_mnist(normalize=True, flatten=True, one_hot_label=False):
    if not os.path.exists(save_file):
        init_mnist()
        
    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)
    
    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
            
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    
    if not flatten:
         for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

    return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label']) 


if __name__ == '__main__':
    init_mnist()
```
	* load_mnist가 데이터셋을 처음 읽을경우 시간이 걸리는데 다음 부터는 pickle파일에 저장 되어서 빠르게 읽음
	* load_mnist 반환값은 **"(훈련이미지, 훈련레이블), (시험 이미지, 시험 레이블)"** 형식으로 됨
	* normalize는 입력값을 0.0 ~ 1.0사이의 값으로 정규화 할지 False는 0~255사이
	* flatten는 입력 이미지를 True는 784개의 1차원 False는 1x28x28의 3차원 배열
	* one_hot_label 
	True: **원-핫 인코딩(정답을 뜻하는 1개의 원소만 1이고 나머지는 0)**형태로 저장 	Flase: 숫자 형태의 레이블을 저장

3.6.2 신경망 추론 처리
	* 입력층 뉴런 784개(28x28), 출력층 뉴런 10개(0~9사이의 숫자)
	* 첫번째 은닉층은 50개, 두번째 은닉층은 100개로 가정
	* 정확도를 평가하는 코드
```python
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax


def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```
	* get_data(): 데이터를 읽음
	* init_network(): 네트워크 생성
	* predict(): 가중치함수

	* 함수의 빌드순서
		1. mnist데이터를 get_data()로 x, t에 치환
		2. 네트워크 활성화
		3. for문을 돌며 predict()함수로 각 레이블의 확률을 넘파이 배열로 반환
		4. np.argmax()로 배열에서 가장 큰 원소의 인덱스를 구함
		5. 신경망이 예측한 확률과 정답 레이블을 비교하여 맞힌 수를 계산
		6. 전체 이미지 숫자를 맞힌 수로 나눠 정확도를 구함
	* 정규화: normalize가 True일경우 0.0~1.0범위로 변환하는 것 처럼 특정범위로 변환     
	* 전처리: 신경망의 입력 데이터에 특정 변환을 가하는 것
	* 위의 소스코드는 입력 이미지 데이터에 대한 전처리 작업으로 정규화 수행

3.6.3 배치 처리
	* 위의 코드를 바탕으로 차원의 원소수는 같음
	* 이미지 데이터 1장의 경우
	X         W1           W2             W3       ->    Y
     784     784*50    50*100   100*10           10
	* 이미지 데이터 여러장의 경우
	X의 값이 Nx784가 고 Y는 100x10이 됨
	100장의 데이터의 결과가 한번에 출력
	* 이처럼 하나로 묶은 입력 데이터를 **배치처리** 라 함

배치 처리 구현
```python
x, t = get_data()
network = init_network()

batch_size = 100 #배치 크기
accuracy_cnt = 0
for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size] 
	  #x[0:100],  x[1:101]등으로 100개씩 묶음
	  y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
	  # 1차원으로 구성된 것들중 큰 값의 인덱스들만 가져옴
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
```
```python
#argmax(x, axis=1)
>>> x = np.array([0.1, 0.8, 0.1],[0.3, 0.1, 0.6],
		  [0.2, 0.5, 0.3],[0.8, 0.1, 0.1])
>>> y = np.argmax(x, axis=1)
>>> print(y)
[1 2 1 0]
```
