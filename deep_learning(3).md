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
