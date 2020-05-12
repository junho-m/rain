#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.

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


# # 체크포인트 훈련하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/checkpoint"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />TensorFlow.org에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/checkpoint.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Google Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/checkpoint.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />깃헙(GitHub) 소스 보기</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs-l10n/site/ko/guide/checkpoint.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />Download notebook</a>
#   </td>
# </table>

# Note: 이 문서는 텐서플로 커뮤니티에서 번역했습니다. 커뮤니티 번역 활동의 특성상 정확한 번역과 최신 내용을 반영하기 위해 노력함에도
# 불구하고 [공식 영문 문서](https://www.tensorflow.org/?hl=en)의 내용과 일치하지 않을 수 있습니다.
# 이 번역에 개선할 부분이 있다면
# [tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n/) 깃헙 저장소로 풀 리퀘스트를 보내주시기 바랍니다.
# 문서 번역이나 리뷰에 참여하려면
# [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)로
# 메일을 보내주시기 바랍니다.

# "텐서플로 모델 저장하기" 라는 문구는 보통 둘중 하나를 의미합니다:
# 
#   1. Checkpoints, 혹은 
#   2. SavedModel.
# 
# Checkpoint는 모델이 사용한 모든 매개변수(`tf.Variable` 객체들)의 정확한 값을 캡처합니다. Chekcpoint는 모델에 의해 정의된 연산에 대한 설명을 포함하지 않으므로 일반적으로 저장된 매개변수 값을 사용할 소스 코드를 사용할 수 있을 때만 유용합니다.
# 
# 반면 SavedModel 형식은 매개변수 값(체크포인트) 외에 모델에 의해 정의된 연산에 대한 일련화된 설명을 포함합니다. 이 형식의 모델은 모델을 만든 소스 코드와 독립적입니다. 따라서 TensorFlow Serving, TensorFlow Lite, TensorFlow.js 또는 다른 프로그래밍 언어(C, C++, Java, Go, Rust, C# 등. TensorFlow APIs)로 배포하기에 적합합니다.
# 
# 이 가이드는 체크포인트 쓰기 및 읽기를 위한 API들을 다룹니다.

# ## 설치

# In[ ]:


import tensorflow as tf


# In[ ]:


class Net(tf.keras.Model):
  """A simple linear model."""

  def __init__(self):
    super(Net, self).__init__()
    self.l1 = tf.keras.layers.Dense(5)

  def call(self, x):
    return self.l1(x)


# In[ ]:


net = Net()


# ## `tf.keras` 훈련 API들로부터 저장하기
# 
# [`tf.keras` 저장하고 복구하는
# 가이드](./keras/overview.ipynb#save_and_restore)를 읽어봅시다.
# 
# `tf.keras.Model.save_weights` 가 텐서플로 CheckPoint를 저장합니다. 

# In[ ]:


net.save_weights('easy_checkpoint')


# ## Checkpoints 작성하기
# 

# 텐서플로 모델의 지속적인 상태는 `tf.Variable` 객체에 저장되어 있습니다. 이들은 직접으로 구성할 수 있지만, `tf.keras.layers` 혹은 `tf.keras.Model`와 같은 고수준 API들로 만들어 지기도 합니다.
# 
# 변수를 관리하는 가장 쉬운 방법은 Python 객체에 변수를 연결한 다음 해당 객체를 참조하는 것입니다. 
# 
# `tf.train.Checkpoint`, `tf.keras.layers.Layer`, and `tf.keras.Model`의 하위클래스들은 해당 속성에 할당된 변수를 자동 추적합니다. 다음 예시는 간단한 선형 model을 구성하고, 모든 model 변수의 값을 포합하는 checkpoint를 씁니다.

# `Model.save_weights`를 사용해 손쉽게 model-checkpoint를 저장할 수 있습니다.

# ### 직접 Checkpoint작성하기

# #### 설치

# `tf.train.Checkpoint`의 모든 특성을 입증하기 위해서 toy dataset과 optimization step을 정의해야 합니다.

# In[ ]:


def toy_dataset():
  inputs = tf.range(10.)[:, None]
  labels = inputs * 5. + tf.range(5.)[None, :]
  return tf.data.Dataset.from_tensor_slices(
    dict(x=inputs, y=labels)).repeat(10).batch(2)


# In[ ]:


def train_step(net, example, optimizer):
  """Trains `net` on `example` using `optimizer`."""
  with tf.GradientTape() as tape:
    output = net(example['x'])
    loss = tf.reduce_mean(tf.abs(output - example['y']))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(gradients, variables))
  return loss


# #### Checkpoint객체 생성
# 
# 인위적으로 checkpoint를 만드려면 `tf.train.Checkpoint` 객체가 필요합니다. Checkpoint하고 싶은 객체의 위치는 객체의 특성으로 설정이 되어 있습니다.
# 
#  `tf.train.CheckpointManager`도 다수의 checkpoint를 관리할때 도움이 됩니다

# In[ ]:


opt = tf.keras.optimizers.Adam(0.1)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)


# #### 훈련하고 model checkpoint작성하기

# 다음 훈련 루프는 model과 optimizer의 인스턴스를 만든 후 `tf.train.Checkpoint` 객체에 수집합니다. 이것은 각 데이터 배치에 있는 루프의 훈련 단계를 호출하고, 주기적으로 디스크에 checkpoint를 작성합니다.

# In[ ]:


def train_and_checkpoint(net, manager):
  ckpt.restore(manager.latest_checkpoint)
  if manager.latest_checkpoint:
    print("Restored from {}".format(manager.latest_checkpoint))
  else:
    print("Initializing from scratch.")

  for example in toy_dataset():
    loss = train_step(net, example, opt)
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 10 == 0:
      save_path = manager.save()
      print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
      print("loss {:1.2f}".format(loss.numpy()))


# In[ ]:


train_and_checkpoint(net, manager)


# #### 복구하고 훈련 계속하기

# 첫 번째 과정 이후 새로운 model과 매니저를 전달할 수 있지만, 일을 마무리 한 정확한 지점에서 훈련을 가져와야 합니다:

# In[ ]:


opt = tf.keras.optimizers.Adam(0.1)
net = Net()
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=net)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=3)

train_and_checkpoint(net, manager)


# `tf.train.CheckpointManager` 객체가 이전 checkpoint들을 제거합니다. 위는 가장 최근의 3개 checkpoint만 유지하도록 구성되어 있습니다.

# In[ ]:


print(manager.checkpoints)  # 남은 checkpoint들 나열


# 예를 들어, `'./tf_ckpts/ckpt-10'`같은 경로들은 디스크에 있는 파일이 아닙니다. 대신에 이 경로들은 `index` 파일과 변수 값들을 담고있는 파일들의 전위 표기입니다. 이 전위 표기들은 `CheckpointManager` 가 상태를 저장하는 하나의 checkpoint 파일 (`'./tf_ckpts/checkpoint'`) 에 그룹으로 묶여있습니다.

# In[ ]:


get_ipython().system('ls ./tf_ckpts')


# <a id="loading_mechanics"/>
# 
# ## 작동 원리
# 
# 텐서플로는 로드되는 객체에서 시작하여 명명된 엣지가 있는 방향 그래프를 통과시켜 변수를 checkpoint된 값과 일치시킵니다. 엣지의 이름들은 특히 기여한 객체의 이름에서 따왔습니다. 예를들면, `self.l1 = tf.keras.layers.Dense(5)`안의 `"l1"`. `tf.train.Checkpoint` 이것의 키워드 전달인자 이름을 사용했습니다, 여기에서는 `"step"` in `tf.train.Checkpoint(step=...)`.
# 
# 위의 예에서 나온 종속성 그래프는 다음과 같습니다.:
# 
# ![훈련 반복 예시의 의존 그래프 시각화](http://tensorflow.org/images/guide/whole_checkpoint.svg)
# 
# optimizer는 빨간색으로, regular 변수는 파란색으로, optimizer 슬롯 변수는 주황색으로 표시합니다. 다른 nodes는, 예를 들면  `tf.train.Checkpoint`, 이 검은색임을 나타냅니다.
# 
# 슬롯 변수는 optimizer의 일부지만 특정 변수에 대해 생성됩니다.  `'m'`  위의 엣지는 모멘텀에 해당하며, 아담 optimizer는 각 변수에 대해 추적합니다. 슬롯 변수는 변수와 optimizer가 모두 저장될 경우에만 checkpoint에 저장되며, 따라서 파선 엣지가 됩니다.

# `tf.train.Checkpoint`로 불러온 `restore()` 오브젝트 큐는그`Checkpoint` 개체에서 일치하는 방법이 있습니다. 변수 값 복원을 요청한 복원 작업 대기 행렬로 정리합니다. 예를 들어, 우리는 네트워크와 계층을 통해 그것에 대한 하나의 경로를 재구성함으로서 위에서 정의한 모델에서 커널만 로드할 수 있습니다.

# In[ ]:


to_restore = tf.Variable(tf.zeros([5]))
print(to_restore.numpy())  # 모두 0입니다.
fake_layer = tf.train.Checkpoint(bias=to_restore)
fake_net = tf.train.Checkpoint(l1=fake_layer)
new_root = tf.train.Checkpoint(net=fake_net)
status = new_root.restore(tf.train.latest_checkpoint('./tf_ckpts/'))
print(to_restore.numpy())  # 우리는 복구된 변수를 이제 얻었습니다.


# 이 새로운 개체에 대한 의존도 그래프는 우리가 위에 적은 더 큰 checkpoint보다 작은 하위 그래프입니다. 이것은 오직  `tf.train.Checkpoint`에서 checkpoints 셀때 편향과 저장 카운터만 포함합니다.
# 
# ![편향 변수의 서브그래프 시각화](http://tensorflow.org/images/guide/partial_checkpoint.svg)
# 
# `restore()` 함수는 선택적으로 확인을 거친 객체의 상태를 반환합니다.  새로 만든 checkpoint에서 우리가 만든 모든 개체가 복원되어 status.assert_existing_objects_match()가 통과합니다.

# In[ ]:


status.assert_existing_objects_matched()


# checkpoint에는 계층의 커널과 optimizer의 변수를 포함하여 일치하지 않는 많은 개체가 있습니다. status.assert_consumed()는 checkpoint와 프로그램이 정확히 일치할 경우에만 통과하고 여기에 예외를 둘 것입니다.

# ### 복구 지연
#  텐서플로우의 Layer 객체는 입력 형상을 이용할 수 있을 때 변수 생성을 첫 번째 호출로 지연시킬 수 있습니다. 예를 들어, 'Dense' 층의 커널의 모양은 계층의 입력과 출력 형태 모두에 따라 달라지기 때문에, 생성자 인수로 필요한 출력 형태는 그 자체로 변수를 만들기에 충분한 정보가 아닙니다. 예를 들어, 'Dense' 층의 커널의 모양은 계층의 입력과 출력 형태 모두에 따라 달라지기 때문에, 생성자 인수로 필요한 출력 형태는 그 자체로 변수를 만들기에 충분한 정보가 아닙니다.
# 
# 이 관용구를 지지하려면 `tf.train.Checkpoint` queues는 일치하는 변수가 없는 것들을 복원합니다.  

# In[ ]:


delayed_restore = tf.Variable(tf.zeros([1, 5]))
print(delayed_restore.numpy())  # 아직 복원이 안되어 값이 0입니다.
fake_layer.kernel = delayed_restore
print(delayed_restore.numpy())  # 복원되었습니다.


# ### checkpoints 수동 검사
# 
# `tf.train.list_variables`에는 checkpoint 키와 변수 형태가 나열돼있습니다. Checkpoint의 키들은 위에 있는 그래프의 경로입니다.

# In[ ]:


tf.train.list_variables(tf.train.latest_checkpoint('./tf_ckpts/'))


# ### 목록 및 딕셔너리 추적
# 
# `self.l1 = tf.keras.layer.Dense(5)`,와 같은 직접적인 속성 할당은 목록과 사전적 속성에 할당하면 내용이 추적됩니다. 

# In[ ]:


save = tf.train.Checkpoint()
save.listed = [tf.Variable(1.)]
save.listed.append(tf.Variable(2.))
save.mapped = {'one': save.listed[0]}
save.mapped['two'] = save.listed[1]
save_path = save.save('./tf_list_example')

restore = tf.train.Checkpoint()
v2 = tf.Variable(0.)
assert 0. == v2.numpy()  # 아직 복구되지 않았습니다.
restore.mapped = {'two': v2}
restore.restore(save_path)
assert 2. == v2.numpy()


# 당신은 래퍼(wrapper) 객체를 목록과 사전에 있음을 알아차릴겁니다. 이러한 래퍼는 기본 데이터 구조의 checkpoint 가능한 버전입니다. 속성 기반 로딩과 마찬가지로, 이러한 래퍼들은 변수의 값이 용기에 추가되는 즉시 복원됩니다. 

# In[ ]:


restore.listed = []
print(restore.listed)  # 리스트래퍼([])
v1 = tf.Variable(0.)
restore.listed.append(v1)  # 이전 셀의 restore()에서 v1 복원합니다.
assert 1. == v1.numpy()


# f.keras의 하위 클래스에 동일한 추적이 자동으로 적용되고 예를 들어 레이어 목록을 추적하는 데 사용할 수 있는 모델입니다.

# ##Estimator를 사용하여 객체 기반 checkpoint를 저장하기
# 
#  [Estimator 가이드](https://www.tensorflow.org/guide/estimator)를 보십시오.
# 
# Estimators는 기본적으로 이전 섹션에서 설명한 개체 그래프 대신 변수 이름을 가진 체크포인트를 저장합니다. tf.train.Checkpoint는 이름 기반 체크포인트를 사용할 수 있지만, 모델의 일부를 Estimator's model_fn 외부로 이동할 때 변수 이름이 변경될 수 있습니다. 객체 기반 checkpoints를 저장하면 Estimator 내에서 모델을 훈련시킨 후 외부에서 쉽게 사용할 수 있습니다.

# In[ ]:


import tensorflow.compat.v1 as tf_compat


# In[ ]:


def model_fn(features, labels, mode):
  net = Net()
  opt = tf.keras.optimizers.Adam(0.1)
  ckpt = tf.train.Checkpoint(step=tf_compat.train.get_global_step(),
                             optimizer=opt, net=net)
  with tf.GradientTape() as tape:
    output = net(features['x'])
    loss = tf.reduce_mean(tf.abs(output - features['y']))
  variables = net.trainable_variables
  gradients = tape.gradient(loss, variables)
  return tf.estimator.EstimatorSpec(
    mode,
    loss=loss,
    train_op=tf.group(opt.apply_gradients(zip(gradients, variables)),
                      ckpt.step.assign_add(1)),
    # Estimator가 "ckpt"를 객체 기반의 꼴로 저장하게 합니다.
    scaffold=tf_compat.train.Scaffold(saver=ckpt))

tf.keras.backend.clear_session()
est = tf.estimator.Estimator(model_fn, './tf_estimator_example/')
est.train(toy_dataset, steps=10)


# `tf.train.Checkpoint`는 그런 다음 `model_dir`에서 Estimator의 checkpoints를 로드할 수 있습니다.

# In[ ]:


opt = tf.keras.optimizers.Adam(0.1)
net = Net()
ckpt = tf.train.Checkpoint(
  step=tf.Variable(1, dtype=tf.int64), optimizer=opt, net=net)
ckpt.restore(tf.train.latest_checkpoint('./tf_estimator_example/'))
ckpt.step.numpy()  # est.train(..., steps=10)부터


# ## 요약
# 
# 텐서프로우 객체는 사용하는 변수의 값을 저장하고 복원할 수 있는 쉬운 자동 메커니즘을 제공합니다.
# 
