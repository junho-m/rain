#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.
# 

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


# # GPU 사용하기
# 
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/gpu"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />TensorFlow.org에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/gpu.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/gpu.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />깃허브(GitHub) 소스 보기</a>
#   </td>
# </table>

# Note: 이 문서는 텐서플로 커뮤니티에서 번역했습니다. 커뮤니티 번역 활동의 특성상 정확한 번역과 최신 내용을 반영하기 위해 노력함에도
# 불구하고 [공식 영문 문서](https://www.tensorflow.org/?hl=en)의 내용과 일치하지 않을 수 있습니다.
# 이 번역에 개선할 부분이 있다면
# [tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n/) 깃헙 저장소로 풀 리퀘스트를 보내주시기 바랍니다.
# 문서 번역이나 리뷰에 참여하려면
# [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)로
# 메일을 보내주시기 바랍니다.

# ## 설정
# 
# 최신 버전의 텐서플로가 설치되어있는지 확인하세요.

# In[ ]:


import tensorflow as tf


# ## 장치 할당 로깅
# 
# 연산과 텐서가 어떤 장치에 할당되었는지 확인하려면 `tf.debugging.set_log_device_placement(True)`를 프로그램의 가장 처음에 선언하세요. 장치 할당 로깅을 활성화하면 모든 텐서나 연산 할당이 출력됩니다.

# In[ ]:


tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


# 위 코드는 `MatMul` 연산이 `GPU:0`에서 수행되었다고 보여줄 것입니다.

# ## 장치 수동 할당
# 
# 특정 연산을 수행할 장치를 직접 선택하고 싶다면, `with tf.device`로 장치 컨텍스트를 생성할 수 있고 해당 컨텍스트에서의 모든 연산은 지정된 장치에서 수행됩니다.

# In[ ]:


tf.debugging.set_log_device_placement(True)

# 텐서를 CPU에 할당
with tf.device('/CPU:0'):
  a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
  b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

c = tf.matmul(a, b)
print(c)


# `a`와 `b`가 `CPU:0`에 할당되었습니다. `MatMul` 연산은 수행할 장치가 명시적으로 할당되어 있지 않기 때문에 텐서플로 런타임(runtime)은 연산과 가용한 장치들(이 예제에서는 `GPU:0`)을 기반으로 하나를 고를 것이고 필요하다면 장치들간에 텐서를 자동으로 복사할 것입니다.

# ## GPU 메모리 제한하기
# 
# 기본적으로 텐서플로는 모든 GPU의 거의 모든 메모리를 프로세스가 볼 수 있도록 매핑합니다([`CUDA_VISIBLE_DEVICES`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#env-vars)에 포함되었다고 가정합니다). 이는 메모리 단편화를 줄여서 상대적으로 귀한 GPU 메모리 리소스를 장치에서 보다 효율적으로 사용할 수 있게 합니다. `tf.config.experimental.set_visible_devices` 메서드를 사용하여 텐서플로에서 접근할 수 있는 GPU를 조정할 수 있습니다.

# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
    print(e)


# 어떤 경우에는 프로세스가 가용한 메모리의 일부에만 할당되도록 하거나 프로세스의 요구량만큼 메모리 사용이 가능할 필요가 있습니다. 텐서플로에서는 이를 위해 두 가지 방법을 제공합니다.
# 
# 첫 번째 방법은 `tf.config.experimental.set_memory_growth`를 호출하여 메모리 증가를 허용하는 것입니다. 이는 런타임에서 할당하는데 필요한 양만큼의 GPU 메모리를 할당합니다: 처음에는 메모리를 조금만 할당하고, 프로그램이 실행되어 더 많은 GPU 메모리가 필요하면, 텐서플로 프로세스에 할당된 GPU 메모리 영역을 확장합니다. 메모리 해제는 메모리 단편화를 악화시키므로 메모리 해제는 하지 않습니다. 특정 GPU의 메모리 증가를 허용하려면 다음 코드를 텐서나 연산 앞에 입력하세요.

# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
  except RuntimeError as e:
    # 프로그램 시작시에 메모리 증가가 설정되어야만 합니다
    print(e)


# 또 다른 방법은 `TF_FORCE_GPU_ALLOW_GROWTH` 환경변수를 `true`로 설정하는 것입니다. 이 설정은 플랫폼 종속적입니다.
# 
# 두 번째 방법은 `tf.config.experimental.set_virtual_device_configuration`으로 가상 GPU 장치를 설정하고 GPU에 할당될 전체 메모리를 제한하는 것입니다.

# In[ ]:


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # 텐서플로가 첫 번째 GPU에 1GB 메모리만 할당하도록 제한
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
  except RuntimeError as e:
    # 프로그램 시작시에 가상 장치가 설정되어야만 합니다
    print(e)


# 이는 텐서플로 프로세스에서 사용가능한 GPU 메모리량을 제한하는데 유용합니다. 워크스테이션 GUI같이 GPU가 다른 어플리케이션들에 공유되는 로컬 개발환경에서 보통 사용되는 방법입니다.

# ## 멀티 GPU 시스템에서 하나의 GPU만 사용하기
# 
# 시스템에 두 개 이상의 GPU가 있다면 낮은 ID의 GPU가 기본으로 선택됩니다. 다른 GPU에서 실행하고 싶으면 명시적으로 표시해야 합니다:

# In[ ]:


tf.debugging.set_log_device_placement(True)

try:
  # 유효하지 않은 GPU 장치를 명시
  with tf.device('/device:GPU:2'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
except RuntimeError as e:
  print(e)


# 명시한 장치가 존재하지 않으면 `RuntimeError`가 나옵니다:
# 
# 명시한 장치가 존재하지 않을 때 텐서플로가 자동으로 현재 지원하는 장치를 선택하게 하려면 `tf.config.set_soft_device_placement(True)`를 호출하세요.

# In[ ]:


tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

# 텐서 생성
a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)

print(c)


# ## 멀티 GPU 사용하기

# #### `tf.distribute.Strategy` 사용
# 
# 멀티 GPU를 사용하는 가장 좋은 방법은 `tf.distriute.Strategy`를 사용하는 것입니다. 간단한 예제를 살펴봅시다:

# In[ ]:


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
  inputs = tf.keras.layers.Input(shape=(1,))
  predictions = tf.keras.layers.Dense(1)(inputs)
  model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
  model.compile(loss='mse',
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.2))


# 이 프로그램은 입력 데이터를 나누고 모델의 복사본을 각 GPU에서 실행할 것입니다. 이는 "[데이터 병렬처리](https://en.wikipedia.org/wiki/Data_parallelism)"라고도 합니다.
# 
# 병렬화 전략에 대해 더 알고 싶으시면 [가이드](./distributed_training.ipynb)를 참조하세요.

# #### `tf.distribute.Strategy` 미사용
# 
# `tf.distribute.Strategy`는 여러 장치에 걸쳐 계산을 복제해서 동작합니다. 모델을 각 GPU에 구성하여 수동으로 이를 구현할 수 있습니다. 예를 들면:

# In[ ]:


tf.debugging.set_log_device_placement(True)

gpus = tf.config.experimental.list_logical_devices('GPU')
if gpus:
  # 여러 GPU에 계산을 복제
  c = []
  for gpu in gpus:
    with tf.device(gpu.name):
      a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
      b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
      c.append(tf.matmul(a, b))

  with tf.device('/CPU:0'):
    matmul_sum = tf.add_n(c)

  print(matmul_sum)

