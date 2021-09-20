# 퍼셉트론
- - - -
다수의 신호를 입력 받아 하나의 신호 출력
여기서 신호란 전류나 강물처럼 흐름이 있는 것으로 여김
각각의 입력신호에 고유한 가중치를 곱해 신호의 총합이 정해진 한계를 넘으면 1출력
한계를 임계값이라 부름
기호로 $\theta$ 세타를 사용
$$
y =
    \begin{cases}
      0 & \text{w1x1 + w2x2<= $\theta$}\\
      1 & \text{w1x1 + w2x2 > $\theta$}
    \end{cases}    
$$
- - - -
단순 논리 회로
AND 게이트
x1 x2 y
0   0  0
1   0  0
0   1  0
1   1  1

이때의 (w1,w2, &\theta&) 의 값은 (0.5, 0.5, 0.8) 등 이 있다
이렇게 하면 x1, y1이 모두 1일때만 출력
AND외에도 NAND, OR 게이트도 표현할 수 있다
- - - -
퍼셉트론 구현
```python
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
	  tmp = x1*w1 + x2*w2
	  if tmp <= theta:
		  return 0
	  elif tmp > theta:
		  return 1

AND(0, 0) #0
AND(1, 0) #0
AND(0, 1) #0
AND(1, 1) #1
```

가중치와 편향 도입
$$
y =
    \begin{cases}
      0 & \text{b + w1x1 + w2x2<= 0}\\
      1 & \text{b + w1x1 + w2x2 > 0}
    \end{cases}    
$$
위에서 -$\theta$가 b로 변함
b는 편향
```python
>>> import numpy as np
>>> x = np.array([0, 1]) #입력
>>> y = np.array([0.5, 0.5]) #가중치
>>> b = -0.7
>>> w*x
array([0., 0.5])
>>> np.sum(w*x)
0.5
>>> np.sum(w*x) + b
-0.1999999 # 대략 -0.2
```

가중치와 편향 구현
```python
def AND(x1, x2):
	  x = np.array([x1, x2])
	  w = np.array([0.5, 0.5])
	  b = -0.7
	  tmp = np.sum(w*x) + b
	  if tmp <= 0:
		  return 0
	  else:
		  return 1
```
가중치는 입력 신호가 결과에 주는 영향력(중요도)를 조절하는 매개변수
편향은 뉴런이 얼마나 쉽게 활성화 하느냐를 조정하는 매개변수
예를 들어
b가 -0.1이면 각 입력 신호에 가중치를 곱해 0.1만 넘기면 되는데
b가 -20.0이면 각 입력 신호에 가중치를 곱해 20.0을 넘어야되므로
편향의 값은 얼마나 쉽게 활성화 되는지 결정
각각을 구분하기도 하지만 문맥에 따라 셋다 가중치라고도 함

NAND, OR 게이트 구현
```python
def NAND(x1, x2):
	  x = np.array([x1, x2])
	  w = np.array([-0.5, -0.5]) #AND와 가중치와 편향만 다름
	  b = 0.7
	  tmp = np.sum(w*x) + b
	  if tmp <= 0:
		  return 0
	  else:
		  return 1

def OR(x1, x2):
	  x = np.array([x1, x2])
	  w = np.array([0.5, 0.5])
	  b = -0.2
	  tmp = np.sum(w*x) + b
	  if tmp <= 0:
		  return 0
	  else:
		  return 1
```

퍼셉트론으로는 XOR게이트를 구현할 수 없음
대신 층을 쌓아 올리는 다층 퍼셉트론을 사용가능
(x1 nand x2) and (x1 or x2) == XOR 게이트

XOR게이트 구현
```python
def XOR(x1, x2):
    s1 = NAND(x1, x2)
	  s2 = OR(x1, x2)
	  y = AND(s1, s2)
    return y

XOR(0, 0) #0
XOR(0, 1) #1
XOR(1, 0) #1
XOR(1, 1) #0
```
