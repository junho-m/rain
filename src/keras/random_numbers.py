#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2019 The TensorFlow Authors.

# In[ ]:


#@title Licensed under the Apache License, Version 2.0 (the "License");
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


# # 난수 생성

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/random_numbers"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />TensorFlow.org에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/random_numbers.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/random_numbers.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />깃허브(GitHub)에서 소스 보기</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs-l10n/site/ko/guide/random_numbers.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />노트북 다운로드하기</a>
#   </td>
# </table>

# Note: 이 문서는 텐서플로 커뮤니티에서 번역했습니다. 커뮤니티 번역 활동의 특성상 정확한 번역과 최신 내용을 반영하기 위해 노력함에도
# 불구하고 [공식 영문 문서](https://www.tensorflow.org/?hl=en)의 내용과 일치하지 않을 수 있습니다.
# 이 번역에 개선할 부분이 있다면
# [tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n/) 깃헙 저장소로 풀 리퀘스트를 보내주시기 바랍니다.
# 문서 번역이나 리뷰에 참여하려면
# [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)로
# 메일을 보내주시기 바랍니다.

# 텐서플로는 `tf.random` 모듈에서 유사 난수 생성기(pseudo random number generator, RNG)를 제공 합니다. 이 문서에서는 난수 생성기를 다루는 방법과 이 기능이 어떻게 다른 텐서플로의 서브 시스템과 상호작용 하는지 설명합니다. 
# 
# 텐서플로는 난수 생성 프로세스를 다루기 위한 두 가지 방식을 제공합니다:
# 
# 1. `tf.random.Generator` 객체 사용을 통한 방식. 각 객체는 상태를 (`tf.Variable` 안에) 유지합니다. 이 상태는 매 숫자 생성때마다 변하게 됩니다.
# 
# 2. `tf.random.stateless_uniform`와 같은 순수-함수형 무상태 랜덤 함수를 통한 방식. 같은 디바이스에서 동일한 인수를 (시드값 포함) 통해 해당 함수를 호출 하면 항상 같은 결과를 출력 합니다.
# 
# 주의: `tf.random.uniform` 와 `tf.random.normal` 같은 구버전 TF 1.x의 RNG들은 아직 삭제되지 않았지만 사용을 권장하지 않습니다.
# 
# 주의: 랜덤 함수는 텐서 플로 버전에 따라 동일함을 보장하지 않습니다. 참조: [버전 호환성](https://www.tensorflow.org/guide/versions#what_is_not_covered)

# ## 설정

# In[ ]:


import tensorflow as tf

# 분산 전략을 위해 2개의 가상 디바이스 cpu:0 and cpu:1 를 생성 합니다.
physical_devices = tf.config.experimental.list_physical_devices("CPU")
tf.config.experimental.set_virtual_device_configuration(
    physical_devices[0], [
        tf.config.experimental.VirtualDeviceConfiguration(),
        tf.config.experimental.VirtualDeviceConfiguration()
    ])


# ## `tf.random.Generator` 클래스
# 
# 각 RNG 호출마다 다른 결과를 생성하기 원할 경우 `tf.random.Generator` 클래스를 사용할 수 있습니다. 이는 내부 상태를 (`tf.Variable` 객체가 관리) 유지 합니다. 이 상태는 난수가 생성될때마다 업데이트 됩니다. 상태가 `tf.Variable`에 의해 유지되기 때문에, 쉬운 체크포인팅(checkpointing), 자동적인 컨트롤 종속, 쓰레드(thread) 안전성의 장점이 있습니다.
# 
# 클래스를 통해서 객체를 직접 생성하거나 전역 생성기를 반환하는 `tf.random.get_global_generator()` 를 호출함을 통해서 `tf.random.Generator` 를 사용할 수 있습니다. :

# In[ ]:


g1 = tf.random.Generator.from_seed(1)
print(g1.normal(shape=[2, 3]))
g2 = tf.random.get_global_generator()
print(g2.normal(shape=[2, 3]))


# 객체를 생성하는데에는 여러 방법이 있습니다. 위에서 볼 수 있는것 처럼 가장 쉬운 방법은 `Generator.from_seed`를 사용하는 것 이며 이는 시드를 통해서 생성기를 생성 합니다. 시드는 0 이상의 정수형 입니다. `from_seed`는 `alg`를 추가적으로 전달 받을 수 있으며 이는 생성기가 사용할 RNG 알고리즘 입니다:

# In[ ]:


g1 = tf.random.Generator.from_seed(1, alg='philox')
print(g1.normal(shape=[2, 3]))


# 추가적인 정보는 *알고리즘* 섹션을 참고해 주세요.
# 
# 생성기를 생성하는 다른 방법은 `Generator.from_non_deterministic_state`를 사용하는 것 입니다. 이 방법을 통해서 생성된 생성기는 비 결정 상태에서 시작 합니다. 이 상태는 시간과 운영 체제 등에 영향을 받습니다.

# In[ ]:


g = tf.random.Generator.from_non_deterministic_state()
print(g.normal(shape=[2, 3]))


# 명백한 상태(explicit state)에서 생성기를 생성하는 방법 등 생성기를 초기화하는 방법은 여러가지가 있지만 해당 가이드에서는 다루지 않습니다.
# 
# 전역 생성기를 사용하기 위해서 `tf.random.get_global_generator`를 사용할 경우, 디바이스 환경에 유의해야 합니다. 전역 생성기는 `tf.random.get_global_generator`가 처음 호출될때 (비 결정 상태로) 생성 되고 기본 디바이스에 배치 됩니다. 예를 들어서 `tf.random.get_global_generator`를 `tf.device("gpu")` 영역에서 처음으로 호출하였을 경우, 전역 생성기는 GPU에 할당 되며, 추후에 CPU에서 사용할시에 GPU-CPU간 복제를 하게 됩니다.
# 
# 생성기를 다른 객체로 변경하기 위한 `tf.random.set_global_generator` 함수도 있습니다. 이 함수는 조심히 사용해야합니다. `tf.function`가 이전의 생성기를 (약한 참조로) 사용하고 있을 수 있으며, 이를 변경하는 것은 가비지 콜렉션(garbage collection)을 발생시켜 `tf.function`에 문제를 유발할 수 있습니다. 전역 생성기를 재설정 하는데에 더 좋은 방법은 `Generator.reset_from_seed` 와 같이 새로운 생성기를 생성하지 않는 "리셋" 함수를 사용하는 것 입니다.

# In[ ]:


g = tf.random.Generator.from_seed(1)
print(g.normal([]))
print(g.normal([]))
g.reset_from_seed(1)
print(g.normal([]))


# ### 독립적인 난수 스트림 생성
# 
# 많은 어플리케이션에서는 서로 겹치지 않으며 통계적으로 상관관계를 가지지 않는 여러개의 독립적인 난수 스트림이 필요 합니다. 이는 각각 독립이 보장된 여러 생성기를 생성 하는 `Generator.split`를 통해서 해결 할 수 있습니다 (즉 독립 스트림 생성할 수 있습니다).

# In[ ]:


g = tf.random.Generator.from_seed(1)
print(g.normal([]))
new_gs = g.split(3)
for new_g in new_gs:
  print(new_g.normal([]))
print(g.normal([]))


# `split`은 RNG의 `normal`과 같이 생성기의 (위의 예제에서는 `g`) 상태를 변경 합니다. 서로 독립인것과 더불어 새로운 생성기는 (`new_gs`) 또한 이전 생성기와 독립임을 보장 합니다 (`g`).
# 
# 새로운 생성기를 생성 하는 것은 디바이스간 복제의 오버헤드를 피하기 위해 사용하고 있는 생성기가 서로 다른 연산에서 동일한 디바이스에 있음을 확실히 하고 싶을때 유용합니다. 예를 들어: 

# In[ ]:


with tf.device("cpu"):  # "cpu"를 사용하고 싶은 디바이스로 변경
  g = tf.random.get_global_generator().split(1)[0]  
  print(g.normal([]))  # 전역 생성기와는 다르게 g를 사용하는 것은 디바이스간 복제를 발생하지 않습니다.


# 참고: 이론적으로, `split` 대신에 `from_seed` 생성자(Constructor)를 사용할 수 있습니다. 그러나 이는 새로운 생성기가 전역 생성기에 독립임을 보장하지 않습니다. 또한 두 생성기의 시드가 동일 하거나 랜덤 생성 스트림이 겹치는 시드를 생성하는 등의 위험이 있습니다.
# 
# `split`을 분할(split)된 생성기를 통해서 호출하면 재귀적으로 분할할 수 있습니다. 재귀 깊이의 제한은 없지만, 오버플로우는 방지하도록 되어있습니다.

# ### `tf.function`와의 상호 작용
# 
# `tf.random.Generator`는 `tf.function`와 사용될 경우 `tf.Variable`와 동일한 규칙이 적용 됩니다. 이는 3가지 측면을 가집니다.

# #### `tf.function` 밖에서 생성기 생성하기 
# 
# `tf.function` 는 밖에서 생성된 생성기를 사용 할 수 있습니다.

# In[ ]:


g = tf.random.Generator.from_seed(1)
@tf.function
def foo():
  return g.normal([])
print(foo())


# 사용자는 함수를 호출할때 생성기 객체가 여전히 살아 있음을 확실히 해야 합니다 (가비지 콜렉션이 되지 않아야 합니다).

# #### `tf.function` 안에서 생성기 생성하기
# 
# `tf.function` 안에서 생성기를 생성하는 경우는 오직 함수의 첫번째 호출에서만 실행 됩니다. 

# In[ ]:


g = None
@tf.function
def foo():
  global g
  if g is None:
    g = tf.random.Generator.from_seed(1)
  return g.normal([])
print(foo())
print(foo())


# #### 생성기를 `tf.function`의 파라미터로 보내기
# 
# `tf.function`의 파라미터로 사용될 경우, 동일한 상태 사이즈를 가진 서로 다른 생성기 객체는 `tf.function`를 재추적(retracing) 하지 않습니다 (상태 사이즈는 RNG 알고리즘에 의해 결정됩니다). 반면, 다른 상태 사이즈를 가질 경우에는 동작 합니다.

# In[ ]:


num_traces = 0
@tf.function
def foo(g):
  global num_traces
  num_traces += 1
  return g.normal([])
foo(tf.random.Generator.from_seed(1))
foo(tf.random.Generator.from_seed(2))
print(num_traces)


# ### 분산 전략(distribution strategies)과의 상호 작용 
# 
# `Generator`가 분산 전략과 상호작용 하는 방식은 3가지가 있습니다.

# #### 분산 전략 밖에서 생성기 생성
# 
# 생성기가 전략 스코프(scope) 밖에서 생성될 경우, 생성기에 대한 모든 복제(replicas)의 접근이 직렬화(serialized) 되고 따라서 복제들은 서로 다른 난수를 가지게 됩니다.

# In[ ]:


g = tf.random.Generator.from_seed(1)
strat = tf.distribute.MirroredStrategy(devices=["cpu:0", "cpu:1"])
with strat.scope():
  def f():
    print(g.normal([]))
  results = strat.run(f)


# 이 사용법은 생성기의 디바이스가 복제에 따라 다르기 때문에 성능에 대한 이슈가 있습니다.

# #### 분산 전략(distribution strategies)안에서 생성기 생성하기
# 
# 전략 영역(strategy scopes)안에서 생성기를 생성하는 것은 허용되지 않았습니다. 이는 생성기를 어떻게 복제하는지에 대한 모호함이 있기 때문입니다 (예를 들어 각 복제본이 동일한 난수를 갖도록 복제를 하거나 서로 다른 난수를 갖도록 `split` 복제를 하는지에 대한 모호함이 있습니다).

# In[ ]:


strat = tf.distribute.MirroredStrategy(devices=["cpu:0", "cpu:1"])
with strat.scope():
  try:
    tf.random.Generator.from_seed(1)
  except ValueError as e:
    print("ValueError:", e)


# `Strategy.run`가 파라미터 함수를 전략 영역 안에서 암묵적으로 실행 함을 유의해야합니다. :

# In[ ]:


strat = tf.distribute.MirroredStrategy(devices=["cpu:0", "cpu:1"])
def f():
  tf.random.Generator.from_seed(1)
try:
  strat.run(f)
except ValueError as e:
  print("ValueError:", e)


# #### 생성기를 `Strategy.run`의 파라미터로 사용하기
# 
# 각 복제본이 각자의 생성기를 사용하길 원할 경우, `n`개의 생성기가 필요 합니다 (각각 복제하거나 split). `n`은 복제의 갯수이며 , 이를 `Strategy.run`의 파라미터로 보냅니다.
# 

# In[ ]:


strat = tf.distribute.MirroredStrategy(devices=["cpu:0", "cpu:1"])
gs = tf.random.get_global_generator().split(2)
# to_args는 run함수를 위한 파라미터를 생성하는 API의 대안 입니다. 
# 이는 추후에 API로 지원할 경우 교체될 예정 입니다.
def to_args(gs):  
  with strat.scope():
    def f():
      return [gs[tf.distribute.get_replica_context().replica_id_in_sync_group]]
    return strat.run(f)
args = to_args(gs)
def f(g):
  print(g.normal([]))
results = strat.run(f, args=args)


# ## 무상태 RNGs
# 
# 무상태(stateless) RNGs의 사용법은 간단합니다. 순수 함수이기 때문에 부작용이 없습니다.

# In[ ]:


print(tf.random.stateless_normal(shape=[2, 3], seed=[1, 2]))
print(tf.random.stateless_normal(shape=[2, 3], seed=[1, 2]))


# 모든 무상태 RNG는 `seed` 파라미터를 필요로 합니다. 이 파라미터는 크기가 `[2]`인 정수형 텐서입니다. 연산의 결과는 이 시드값에 의해 결정 됩니다.

# ## 알고리즘

# ### 일반
# 
# `tf.random.Generator` 클래스와 `stateless` 함수는 모든 디바이스에서 필록스(Philox) 알고리즘을 지원 합니다 (`"philox"` 또는 `tf.random.Algorithm.PHILOX`로 명시할 수 있습니다).
# 
# 만약 같은 알고리즘을 쓰고 같은 상태에서 시작할 경우 서로 다른 디바이스는 같은 정수를 생성 합니다. 또한 "거의 같은" 부동 소수점 수를 생성 합니다. 각 디바이스의 부동 소수점 연산의 방식에 따라 약간의 오차가 발생 할 수 있습니다. (예: reduction order).

# ### XLA 디바이스
# 
# XLA 기반 디바이스에서는 (TPU와 XLA가 활성화된 CPU/GPU) ThreeFry 알고리즘을 지원 합니다. (`"threefry"` 또는 `tf.random.Algorithm.THREEFRY`) 이 알고리즘은 TPU에서 빠르고 CPU/GPU에서는 Philox에 비해 느립니다. 

# 이 알고리즘들에 대한 상세한 정보는 ['Parallel Random Numbers: As Easy as 1, 2, 3'](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf) 논문을 참고해 주세요.
