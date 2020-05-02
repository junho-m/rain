#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.
# 
# Licensed under the Apache License, Version 2.0 (the "License");

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License"); { display-mode: "form" }
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# # 텐서플로 2.0의 tf.function과 오토그래프 (AutoGraph)

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/function"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />TensorFlow.org 에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/function.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Google Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/function.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />깃헙(GitHub) 소스 보기</a>
#   </td>
# </table>

# Note: 이 문서는 텐서플로 커뮤니티에서 번역했습니다. 커뮤니티 번역 활동의 특성상 정확한 번역과 최신 내용을 반영하기 위해 노력함에도
# 불구하고 [공식 영문 문서](https://www.tensorflow.org/?hl=en)의 내용과 일치하지 않을 수 있습니다.
# 이 번역에 개선할 부분이 있다면
# [tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n/) 깃헙 저장소로 풀 리퀘스트를 보내주시기 바랍니다.
# 문서 번역이나 리뷰에 참여하려면
# [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)로
# 메일을 보내주시기 바랍니다.

# TF 2.0 버전은 즉시 실행 (eager execution)의 편리함과 TF 1.0의 성능을 합쳤습니다. 이러한 결합의 중심에는 `tf.function` 이 있는데, 이는 파이썬 문법의 일부를 이식 가능하고 높은 성능의 텐서플로 그래프 코드로 변환시켜줍니다. 
# 
# `tf.function`의 멋지고 새로운 특징은 오토그래프 (AutoGraph)입니다. 이는 자연스러운 파이썬 문법을 활용해서 그래프 코드를 작성할 수 있도록 돕습니다. 오토그래프로 사용할 수 있는 파이썬 특징들의 목록을 보려면 [오토그래프 지원 범위](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/autograph/LIMITATIONS.md)를 참고하세요. `tf.function`에 관한 더 자세한 내용을 확인하려면 RFC [TF 2.0: Functions, not Sessions](https://github.com/tensorflow/community/blob/master/rfcs/20180918-functions-not-sessions-20.md)을 참고하세요. 오토그래프에 대한 더 자세한 내용은 `tf.autograph`를 참고하세요.
# 
# 본 튜토리얼은 `tf.function`와 오토그래프의 기초적인 특징에 대해서 설명할 것입니다. 

# ## 설정
# 
# 텐서플로 2.0 프리뷰 나이틀리 (Preview Nightly) 버전을 임포트(import)하고, TF 2.0 모드를 설정합니다:

# In[ ]:


import numpy as np


# In[ ]:


get_ipython().system('pip install tensorflow==2.0.0-beta1')
import tensorflow as tf


# ## `tf.function` 데코레이터
# 
# `tf.function`을 함수에 붙여줄 경우, 여전히 다른 일반 함수들처럼 사용할 수 있습니다. 하지만 그래프 내에서 컴파일 되었을 때는 더 빠르게 실행하고, GPU나 TPU를 사용해서 작동하고, 세이브드모델(SavedModel)로 내보내는 것이 가능해집니다.

# In[ ]:


@tf.function
def simple_nn_layer(x, y):
  return tf.nn.relu(tf.matmul(x, y))


x = tf.random.uniform((3, 3))
y = tf.random.uniform((3, 3))

simple_nn_layer(x, y)


# 데코레이터를 붙인 결과를 확인해보면, 텐서플로 런타임시의 모든 상호작용들을 다룰 수 있다는 것을 알 수 있습니다.

# In[ ]:


simple_nn_layer


# 만일 여러분의 코드가 여러 함수들을 포함하고 있다면, 그것들에 모두 데코레이터를 붙일 필요는 없습니다. 데코레이터가 붙은 함수로부터 호출된 모든 함수들은 그래프 모드에서 동작합니다.

# In[ ]:


def linear_layer(x):
  return 2 * x + 1


@tf.function
def deep_net(x):
  return tf.nn.relu(linear_layer(x))


deep_net(tf.constant((1, 2, 3)))


# 작은 연산들을 많이 포함한 그래프의 경우 함수들은 즉시 실행 코드 (eager code) 보다 더 빠르게 동작합니다. 하지만 무거운 연산들을 조금 포함한 그래프의 경우 (컨볼루션 등), 그렇게 빠른 속도 향상은 기대하기 어렵습니다.
# 

# In[ ]:


import timeit
conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
  return conv_layer(image)

image = tf.zeros([1, 200, 200, 100])
# 데이터 준비 (warm up)
conv_layer(image); conv_fn(image)
print("컨볼루션 즉시 실행:", timeit.timeit(lambda: conv_layer(image), number=10))
print("컨볼루션 함수:", timeit.timeit(lambda: conv_fn(image), number=10))
print("컨볼루션의 성능에는 큰 차이가 없다는 것을 확인할 수 있습니다")


# In[ ]:


lstm_cell = tf.keras.layers.LSTMCell(10)

@tf.function
def lstm_fn(input, state):
  return lstm_cell(input, state)

input = tf.zeros([10, 10])
state = [tf.zeros([10, 10])] * 2
# 데이터 준비 (warm up)
lstm_cell(input, state); lstm_fn(input, state)
print("lstm 즉시 실행:", timeit.timeit(lambda: lstm_cell(input, state), number=10))
print("lstm 함수:", timeit.timeit(lambda: lstm_fn(input, state), number=10))


# ## 파이썬의 제어 흐름 사용하기
# 
# `tf.function` 내에서 데이터 기반 제어 흐름을 사용할 때, 파이썬의 제어 흐름 문을 사용할 수 있고, 오토그래프 기능은 그것들을 모두 적절한 텐서플로 연산으로 변환할 수 있습니다. 예를 들어, `if` 문은 `Tensor`를 기반으로 작동해야할 때 `tf.cond()` 로 변환될 수 있습니다. 
# 
# 아래 예시에서, `x`는 `Tensor`이지만 `if`문이 예상한대로 정상 작동합니다:

# In[ ]:


@tf.function
def square_if_positive(x):
  if x > 0:
    x = x * x
  else:
    x = 0
  return x


print('square_if_positive(2) = {}'.format(square_if_positive(tf.constant(2))))
print('square_if_positive(-2) = {}'.format(square_if_positive(tf.constant(-2))))


# Note: 위의 예시는 스칼라값으로 간단한 조건절을 사용하였습니다. 하지만 실제 코드에서는 <a href="#batching">배치(Batching)</a> 가 주로 사용됩니다.

# 오토그래프는 기본적인 파이썬 문인 `while`, `for`, `if`, `break`, `continue`, `return`과 네스팅(nesting)을 지원합니다. 이는 `Tensor` 표현을 `while`과 `if` 문의 조건 부분에서 사용하거나 `for` 루프에서 `Tensor`를 반복할 수 있다는 것을 의미합니다.

# In[ ]:


@tf.function
def sum_even(items):
  s = 0
  for c in items:
    if c % 2 > 0:
      continue
    s += c
  return s


sum_even(tf.constant([10, 12, 15, 20]))


# 또한 오토그래프는 고급 사용자를 위해 낮은 수준의 API를 제공합니다. 예를 들어, 여러분은 생성된 코드를 확인하기 위해 다음과 같이 작성할 수 있습니다. 

# In[ ]:


print(tf.autograph.to_code(sum_even.python_function))


# 다음은 더 복잡한 제어 흐름의 예시입니다:

# In[ ]:


@tf.function
def fizzbuzz(n):
  msg = tf.constant('')
  for i in tf.range(n):
    if tf.equal(i % 3, 0):
      tf.print('Fizz')
    elif tf.equal(i % 5, 0):
      tf.print('Buzz')
    else:
      tf.print(i)

fizzbuzz(tf.constant(15))


# ## 케라스와 오토그래프
# 
# 오토그래프는 기본적으로 비동적(non-dynamic) 케라스 모델에서 사용 가능합니다. 더 자세한 정보를 원한다면, `tf.keras`를 참고하세요.

# In[ ]:


class CustomModel(tf.keras.models.Model):

  @tf.function
  def call(self, input_data):
    if tf.reduce_mean(input_data) > 0:
      return input_data
    else:
      return input_data // 2


model = CustomModel()

model(tf.constant([-2, -4]))


# ## 부수 효과 (Side effects)
# 
# 즉시 실행 모드 (eager mode)처럼 부수 효과를 사용할 수 있습니다. 예를 들면, `tf.function` 내에 있는 `tf.assign`이나 `tf.print`이 있습니다. 또한 부수 효과들은 작업들이 순서대로 실행된다는 것을 보장하기 위해 필수적인 제어 의존성 (control dependency)을 추가합니다.

# In[ ]:


v = tf.Variable(5)

@tf.function
def find_next_odd():
  v.assign(v + 1)
  if tf.equal(v % 2, 0):
    v.assign(v + 1)


find_next_odd()
v


# ## 디버깅
# 
# `tf.function` 과 오토그래프는 코드를 생성하고 텐서플로 그래프 내에서 해당 코드를 추적함으로써 동작합니다. 이 메커니즘은 아직까지는 `pdb`같은 단계적 (step-by-step) 디버거를 지원하지 않습니다. 하지만 일시적으로 `tf.function` 내에서 즉시 실행 (eager execution)을 가능하게 하는 `tf.config.run_functions_eagerly(True)`을 사용하고 가장 선호하는 디버거를 사용할 수 있습니다:

# In[ ]:


@tf.function
def f(x):
  if x > 0:
    # 여기에 중단점(breakpoint)을 설정해 보세요!
    # 예시:
    #   import pdb
    #   pdb.set_trace()
    x = x + 1
  return x

tf.config.experimental_run_functions_eagerly(True)

# 이제 중단점을 설정하고 디버거 내에서 코드를 실행할 수 있습니다.
f(tf.constant(1))

tf.config.experimental_run_functions_eagerly(False)


# ## 고급 예제: 그래프 내 훈련 루프
# 
# 이전 섹션은 케라스 레이어나 모델 내부에서 오토그래프를 활용할 수 있는 것을 보여주었습니다. 오토그래프 코드 안에서 케라스 모델을 활용할 수도 있습니다.
# 
# 이 예제는 배치 불러오기, 그래디언트 계산, 매개변수 갱신, 검증 정확도 계산, 수렴까지 반복 등 그래프 내에서 수행되는 전체 훈련 과정을 통해 간단한 케라스 모델이 어떻게 MNIST 데이터셋에 훈련되는지 보여줍니다.

# ### 데이터 다운로드

# In[ ]:


def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

def mnist_dataset():
  (x, y), _ = tf.keras.datasets.mnist.load_data()
  ds = tf.data.Dataset.from_tensor_slices((x, y))
  ds = ds.map(prepare_mnist_features_and_labels)
  ds = ds.take(20000).shuffle(20000).batch(100)
  return ds

train_dataset = mnist_dataset()


# ### 모델 정의하기

# In[ ]:


model = tf.keras.Sequential((
    tf.keras.layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10)))
model.build()
optimizer = tf.keras.optimizers.Adam()


# ### 훈련 (training) 루프 정의하기

# In[ ]:


compute_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

compute_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()


def train_one_step(model, optimizer, x, y):
  with tf.GradientTape() as tape:
    logits = model(x)
    loss = compute_loss(y, logits)

  grads = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))

  compute_accuracy(y, logits)
  return loss


@tf.function
def train(model, optimizer):
  train_ds = mnist_dataset()
  step = 0
  loss = 0.0
  accuracy = 0.0
  for x, y in train_ds:
    step += 1
    loss = train_one_step(model, optimizer, x, y)
    if tf.equal(step % 10, 0):
      tf.print('스텝', step, ': 손실', loss, '; 정확도', compute_accuracy.result())
  return step, loss, accuracy

step, loss, accuracy = train(model, optimizer)
print('최종 스텝', step, ': 손실', loss, '; 정확도', compute_accuracy.result())


# ## 배치 (Batching)
# 
# 실제 적용시에 배치 (batch) 는 성능을 위해 필수적입니다. 오토그래프로 변환하기 가장 좋은 코드는 제어 흐름이 _배치_ 수준에서 결정되는 코드입니다. 만일 제어 흐름이 개별적인 _예제 (example)_ 수준에서 결정된다면, 성능을 유지하기 위해서 배치 API들을 사용해야합니다.
# 
# 예를 들어, 파이썬으로 다음과 같은 코드를 작성했다면:
# 

# In[ ]:


def square_if_positive(x):
  return [i ** 2 if i > 0 else i for i in x]


square_if_positive(range(-5, 5))


# 텐서플로에서는 다음과 같이 작성하고 싶을 것입니다. (그리고 다음 코드는 실제로 동작합니다!):
# 

# In[ ]:


@tf.function
def square_if_positive_naive(x):
  result = tf.TensorArray(tf.int32, size=x.shape[0])
  for i in tf.range(x.shape[0]):
    if x[i] > 0:
      result = result.write(i, x[i] ** 2)
    else:
      result = result.write(i, x[i])
  return result.stack()


square_if_positive_naive(tf.range(-5, 5))


# 하지만 이 경우는 아래와 같이 작성할 수도 있습니다:
# 

# In[ ]:


def square_if_positive_vectorized(x):
  return tf.where(x > 0, x ** 2, x)


square_if_positive_vectorized(tf.range(-5, 5))

