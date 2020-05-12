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


# # 기존 코드를 TensorFlow 2.0으로 바꾸기
# 
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/migrate">
#     <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
#     TensorFlow.org에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/migrate.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
#     구글 코랩(Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/migrate.ipynb">
#     <img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />
#     깃허브(GitHub) 소스 보기</a>
#   </td>
# </table>

# Note: 이 문서는 텐서플로 커뮤니티에서 번역했습니다. 커뮤니티 번역 활동의 특성상 정확한 번역과 최신 내용을 반영하기 위해 노력함에도
# 불구하고 [공식 영문 문서](https://www.tensorflow.org/?hl=en)의 내용과 일치하지 않을 수 있습니다.
# 이 번역에 개선할 부분이 있다면
# [tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n/) 깃헙 저장소로 풀 리퀘스트를 보내주시기 바랍니다.
# 문서 번역이나 리뷰에 참여하려면
# [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)로
# 메일을 보내주시기 바랍니다.

# 여전히 텐서플로 1.X 버전의 코드를 수정하지 않고 텐서플로 2.0에서 실행할 수 있습니다(`contrib` 모듈은 제외):
# 
# ```
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# ```
# 
# 하지만 이렇게 하면 텐서플로 2.0에서 제공하는 많은 장점을 활용할 수 없습니다. 이 문서는 성능을 높이면서 코드는 더 간단하고 유지보수하기 쉽도록 업그레이드하는 방법을 안내합니다.
# 
# ## 자동 변환 스크립트
# 
# 첫 번째 단계는 [업그레이드 스크립트](./upgrade.md)를 사용해 보는 것입니다.
# 
# 이는 텐서플로 2.0으로 업그레이드하기 위해 처음에 할 일입니다. 하지만 이 작업이 기존 코드를 텐서플로 2.0 스타일로 바꾸어 주지는 못합니다. 여전히 플레이스홀더(placeholder)나 세션(session), 컬렉션(collection), 그외 1.x 스타일의 기능을 사용하기 위해 `tf.compat.v1` 아래의 모듈을 참조하고 있을 것입니다.
# 
# ## 2.0에 맞도록 코드 수정하기
# 
# 텐서플로 1.x 코드를 텐서플로 2.0으로 변환하는 몇 가지 예를 소개하겠습니다. 이 작업을 통해 성능을 최적화하고 간소화된 API의 이점을 사용할 수 있습니다.
# 
# 각각의 경우에 수정하는 패턴은 다음과 같습니다:

# ### 1. `tf.Session.run` 호출을 바꾸세요.
# 
# 모든 `tf.Session.run` 호출을 파이썬 함수로 바꾸어야 합니다.
# 
# * `feed_dict`와 `tf.placeholder`는 함수의 매개변수가 됩니다.
# * `fetches`는 함수의 반환값이 됩니다.
# 
# 표준 파이썬 디버거 `pdb`를 사용하여 함수의 코드를 라인 단위로 실행하고 디버깅할 수 있습니다.
# 
# 작동 결과에 만족하면 그래프 모드에서 효율적으로 실행할 수 있도록 `tf.function` 데코레이터를 추가합니다. 조금 더 자세한 내용은 [오토그래프 가이드](function.ipynb)를 참고하세요.

# ### 2. 파이썬 객체를 사용하여 변수와 손실을 관리하세요.
# 
# `tf.get_variable` 대신에 `tf.Variable`을 사용하세요.
# 
# 모든 `variable_scope`는 파이썬 객체로 바꿀 수 있습니다. 일반적으로 다음 중 하나가 될 것입니다:
# 
# * `tf.keras.layers.Layer`
# * `tf.keras.Model`
# * `tf.Module`
# 
# 만약 (`tf.Graph.get_collection(tf.GraphKeys.VARIABLES)`처럼) 변수의 리스트가 필요하다면 `Layer`와 `Model` 객체의 `.variables`이나 `.trainable_variables` 속성을 사용하세요.
# 
# `Layer`와 `Model` 클래스는 전역 컬렉션이 필요하지 않도록 몇 가지 다른 속성들도 제공합니다. `.losses` 속성은 `tf.GraphKeys.LOSSES` 컬렉션 대신 사용할 수 있습니다.
# 
# 자세한 내용은 [케라스 가이드](keras.ipynb)를 참고하세요.
# 
# 경고: `tf.compat.v1`의 상당수 기능은 암묵적으로 전역 컬렉션을 사용합니다.

# ### 3. 훈련 루프를 업그레이드하세요.
# 
# 풀려는 문제에 맞는 고수준 API를 사용하세요. 훈련 루프(loop)를 직접 만드는 것보다 `tf.keras.Model.fit` 메서드를 사용하는 것이 좋습니다.
# 
# 고수준 함수는 훈련 루프를 직접 만들 때 놓치기 쉬운 여러 가지 저수준의 세부 사항들을 관리해 줍니다. 예를 들어 자동으로 규제(regularization) 손실을 수집하거나 모델을 호출할 때 `training=True`로 매개변수를 설정해 줍니다.
# 
# ### 4. 데이터 입력 파이프라인을 업그레이드하세요.
# 
# 데이터 입력을 위해 `tf.data` 데이터셋을 사용하세요. 이 객체는 효율적이고 간결하며 텐서플로와 잘 통합됩니다.
# 
# `tf.keras.Model.fit` 메서드에 바로 전달할 수 있습니다.
# 
# ```
# model.fit(dataset, epochs=5)
# ```
# 
# 파이썬에서 직접 반복시킬 수 있습니다:
# 
# ```
# for example_batch, label_batch in dataset:
#     break
# ```

# ## 모델 변환하기
# 
# ### 준비

# In[ ]:


import tensorflow as tf


import tensorflow_datasets as tfds


# ### 저수준 변수와 연산 실행
# 
# 저수준 API를 사용하는 예는 다음과 같습니다:
# 
# * 재사용을 위해 변수 범위(variable scopes)를 사용하기
# * `tf.get_variable`로 변수를 만들기
# * 명시적으로 컬렉션을 참조하기
# * 다음과 같은 메서드를 사용하여 암묵적으로 컬렉션을 참조하기:
# 
#   * `tf.global_variables`
#   * `tf.losses.get_regularization_loss`
# 
# * 그래프 입력을 위해 `tf.placeholder`를 사용하기
# * `session.run`으로 그래프를 실행하기
# * 변수를 수동으로 초기화하기

# #### 변환 전
# 
# 다음 코드는 텐서플로 1.x를 사용한 코드에서 볼 수 있는 패턴입니다.
# 
# ```python
# in_a = tf.placeholder(dtype=tf.float32, shape=(2))
# in_b = tf.placeholder(dtype=tf.float32, shape=(2))
# 
# def forward(x):
#   with tf.variable_scope("matmul", reuse=tf.AUTO_REUSE):
#     W = tf.get_variable("W", initializer=tf.ones(shape=(2,2)),
#                         regularizer=tf.contrib.layers.l2_regularizer(0.04))
#     b = tf.get_variable("b", initializer=tf.zeros(shape=(2)))
#     return W * x + b
# 
# out_a = forward(in_a)
# out_b = forward(in_b)
# 
# reg_loss = tf.losses.get_regularization_loss(scope="matmul")
# 
# with tf.Session() as sess:
#   sess.run(tf.global_variables_initializer())
#   outs = sess.run([out_a, out_b, reg_loss],
#       	        feed_dict={in_a: [1, 0], in_b: [0, 1]})
# 
# ```

# #### 변환 후

# 변환된 코드의 패턴은 다음과 같습니다:
# 
# * 변수는 파이썬 지역 객체입니다.
# * `forward` 함수는 여전히 필요한 계산을 정의합니다.
# * `sess.run` 호출은 `forward` 함수를 호출하는 것으로 바뀝니다.
# * `tf.function` 데코레이터는 선택 사항으로 성능을 위해 추가할 수 있습니다.
# * 어떤 전역 컬렉션도 참조하지 않고 규제를 직접 계산합니다.
# * **세션이나 플레이스홀더를 사용하지 않습니다.**

# In[ ]:


W = tf.Variable(tf.ones(shape=(2,2)), name="W")
b = tf.Variable(tf.zeros(shape=(2)), name="b")

@tf.function
def forward(x):
  return W * x + b

out_a = forward([1,0])
print(out_a)


# In[ ]:


out_b = forward([0,1])

regularizer = tf.keras.regularizers.l2(0.04)
reg_loss = regularizer(W)


# ### `tf.layers` 기반의 모델

# `tf.layers` 모듈은 변수를 정의하고 재사용하기 위해 `tf.variable_scope`에 의존하는 층 함수를 포함합니다.

# #### 변환 전
# ```python
# def model(x, training, scope='model'):
#   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#     x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu,
#           kernel_regularizer=tf.contrib.layers.l2_regularizer(0.04))
#     x = tf.layers.max_pooling2d(x, (2, 2), 1)
#     x = tf.layers.flatten(x)
#     x = tf.layers.dropout(x, 0.1, training=training)
#     x = tf.layers.dense(x, 64, activation=tf.nn.relu)
#     x = tf.layers.batch_normalization(x, training=training)
#     x = tf.layers.dense(x, 10, activation=tf.nn.softmax)
#     return x
# 
# train_out = model(train_data, training=True)
# test_out = model(test_data, training=False)
# ```

# #### 변환 후

# * 층을 단순하게 쌓을 경우엔 `tf.keras.Sequential`이 적합합니다. (복잡한 모델인 경우 [맞춤형 층과 모델](keras/custom_layers_and_models.ipynb)이나 [함수형 API](keras/functional.ipynb)를 참고하세요.)
# * 모델이 변수와 규제 손실을 관리합니다.
# * `tf.layers`에서 `tf.keras.layers`로 바로 매핑되기 때문에 일대일로 변환됩니다.
# 
# 대부분 매개변수는 동일합니다. 다른 부분은 다음과 같습니다:
# 
# * 모델이 실행될 때 각 층에 `training` 매개변수가 전달됩니다.
# * 원래 `model` 함수의 첫 번째 매개변수(입력 `x`)는 사라집니다. 층 객체가 모델 구축과 모델 호출을 구분하기 때문입니다.
# 
# 추가 노트:
# 
# * `tf.contrib`에서 규제를 초기화했다면 다른 것보다 매개변수 변화가 많습니다.
# * 더 이상 컬렉션을 사용하지 않기 때문에 `tf.losses.get_regularization_loss`와 같은 함수는 값을 반환하지 않습니다. 이는 훈련 루프를 망가뜨릴 수 있습니다.

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.04),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))


# In[ ]:


train_out = model(train_data, training=True)
print(train_out)


# In[ ]:


test_out = model(test_data, training=False)
print(test_out)


# In[ ]:


# 훈련되는 전체 변수
len(model.trainable_variables)


# In[ ]:


# 규제 손실
model.losses


# ### 변수와 tf.layers의 혼용

# 기존 코드는 종종 저수준 TF 1.x 변수와 고수준 `tf.layers` 연산을 혼용합니다.

# #### 변경 전
# ```python
# def model(x, training, scope='model'):
#   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
#     W = tf.get_variable(
#       "W", dtype=tf.float32,
#       initializer=tf.ones(shape=x.shape),
#       regularizer=tf.contrib.layers.l2_regularizer(0.04),
#       trainable=True)
#     if training:
#       x = x + W
#     else:
#       x = x + W * 0.5
#     x = tf.layers.conv2d(x, 32, 3, activation=tf.nn.relu)
#     x = tf.layers.max_pooling2d(x, (2, 2), 1)
#     x = tf.layers.flatten(x)
#     return x
# 
# train_out = model(train_data, training=True)
# test_out = model(test_data, training=False)
# ```

# #### 변경 후

# 이런 코드를 변환하려면 이전 예제처럼 층별로 매핑하는 패턴을 사용하세요.
# 
# `tf.variable_scope`는 기본적으로 하나의 층입니다. 따라서 `tf.keras.layers.Layer`로 재작성합니다. 자세한 내용은 이 [문서](keras/custom_layers_and_models.ipynb)를 참고하세요.
# 
# 일반적인 패턴은 다음과 같습니다:
# 
# * `__init__`에서 층에 필요한 매개변수를 입력 받습니다.
# * `build` 메서드에서 변수를 만듭니다.
# * `call` 메서드에서 연산을 실행하고 결과를 반환합니다.

# In[ ]:


# 모델에 추가하기 위해 맞춤형 층을 만듭니다.
class CustomLayer(tf.keras.layers.Layer):
  def __init__(self, *args, **kwargs):
    super(CustomLayer, self).__init__(*args, **kwargs)

  def build(self, input_shape):
    self.w = self.add_weight(
        shape=input_shape[1:],
        dtype=tf.float32,
        initializer=tf.keras.initializers.ones(),
        regularizer=tf.keras.regularizers.l2(0.02),
        trainable=True)

  # call 메서드가 그래프 모드에서 사용되면
  # training 변수는 텐서가 됩니다.
  @tf.function
  def call(self, inputs, training=None):
    if training:
      return inputs + self.w
    else:
      return inputs + self.w * 0.5


# In[ ]:


custom_layer = CustomLayer()
print(custom_layer([1]).numpy())
print(custom_layer([1], training=True).numpy())


# In[ ]:


train_data = tf.ones(shape=(1, 28, 28, 1))
test_data = tf.ones(shape=(1, 28, 28, 1))

# 맞춤형 층을 포함한 모델을 만듭니다.
model = tf.keras.Sequential([
    CustomLayer(input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
])

train_out = model(train_data, training=True)
test_out = model(test_data, training=False)


# 노트:
# 
# * 클래스 상속으로 만든 케라스 모델과 층은 v1 그래프(연산간의 의존성이 자동으로 제어되지 않습니다)와 즉시 실행 모드 양쪽에서 실행될 수 있어야 합니다.
#   * 오토그래프(autograph)와 의존성 자동 제어(automatic control dependency)를 위해 `tf.function()`으로 `call()` 메서드를 감쌉니다.
# 
# * `call` 메서드에 `training` 매개변수를 추가하는 것을 잊지 마세요.
#     * 경우에 따라 이 값은 `tf.Tensor`가 됩니다.
#     * 경우에 따라 이 값은 파이썬 불리언(boolean)이 됩니다.
# 
# * `self.add_weight()`를 사용하여 생성자 메서드나 `def build()` 메서드에서 모델 변수를 만듭니다.
#   * `build` 메서드에서 입력 크기를 참조할 수 있으므로 적절한 크기의 가중치를 만들 수 있습니다.
#   * `tf.keras.layers.Layer.add_weight`를 사용하면 케라스가 변수와 규제 손실을 관리할 수 있습니다.
# 
# * 맞춤형 층 안에 `tf.Tensors` 객체를 포함하지 마세요.
#   * `tf.function`이나 즉시 실행 모드에서 모두 텐서가 만들어지지만 이 텐서들의 동작 방식은 다릅니다.
#   * 상태를 저장하기 위해서는 `tf.Variable`을 사용하세요. 변수는 양쪽 방식에 모두 사용할 수 있습니다.
#   * `tf.Tensors`는 중간 값을 저장하기 위한 용도로만 사용합니다.

# ### Slim & contrib.layers를 위한 노트
# 
# 예전 텐서플로 1.x 코드는 [Slim](https://ai.googleblog.com/2016/08/tf-slim-high-level-library-to-define.html) 라이브러리를 많이 사용합니다. 이 라이브러리는 텐서플로 1.x의 `tf.contrib.layers`로 패키지되어 있습니다. `contrib` 모듈은 더 이상 텐서플로 2.0에서 지원하지 않고 `tf.compat.v1`에도 포함되지 않습니다. Slim을 사용한 코드를 TF 2.0으로 변환하는 것은 `tf.layers`를 사용한 코드를 변경하는 것보다 더 어렵습니다. 사실 Slim 코드는 `tf.layers`로 먼저 변환하고 그 다음 케라스로 변환하는 것이 좋습니다.
# 
# * `arg_scopes`를 삭제하세요. 모든 매개변수는 명시적으로 설정되어야 합니다.
# * `normalizer_fn`과 `activation_fn`를 사용해야 한다면 분리하여 각각 하나의 층으로 만드세요.
# * 분리 합성곱(separable conv) 층은 한 개 이상의 다른 케라스 층으로 매핑합니다(깊이별(depthwise), 점별(pointwise), 분리(separable) 케라스 층).
# * Slim과 `tf.layers`는 매개변수 이름과 기본값이 다릅니다.
# * 일부 매개변수는 다른 스케일(scale)을 가집니다.
# * 사전 훈련된 Slim 모델을 사용한다면 `tf.keras.applications`나 [TFHub](https://tensorflow.orb/hub)를 확인해 보세요.
# 
# 일부 `tf.contrib` 층은 텐서플로 내부에 포함되지 못했지만 [TF 애드온(add-on) 패키지](https://github.com/tensorflow/addons)로 옮겨졌습니다.

# ## 훈련

# 여러 가지 방법으로 `tf.keras` 모델에 데이터를 주입할 수 있습니다. 파이썬 제너레이터(generator)와 넘파이 배열을 입력으로 사용할 수 있습니다.
# 
# `tf.data` 패키지를 사용하여 모델에 데이터를 주입하는 것이 권장되는 방법입니다. 이 패키지는 데이터 조작을 위한 고성능 클래스들을 포함하고 있습니다.
# 
# `tf.queue`는 데이터 구조로만 지원되고 입력 파이프라인으로는 지원되지 않습니다.

# ### 데이터셋 사용하기

# [텐서플로 데이터셋(Datasets)](https://tensorflow.org/datasets) 패키지(`tfds`)는 `tf.data.Dataset` 객체로 정의된 데이터셋을 적재하기 위한 유틸리티가 포함되어 있습니다.
# 
# 예를 들어 `tfds`를 사용하여 MNIST 데이터셋을 적재하는 코드는 다음과 같습니다:

# In[ ]:


datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
mnist_train, mnist_test = datasets['train'], datasets['test']


# 그 다음 훈련용 데이터를 준비합니다:
# 
#   * 각 이미지의 스케일을 조정합니다.
#   * 샘플의 순서를 섞습니다.
#   * 이미지와 레이블(label)의 배치를 만듭니다.

# In[ ]:


BUFFER_SIZE = 10 # 실전 코드에서는 더 큰 값을 사용합니다.
BATCH_SIZE = 64
NUM_EPOCHS = 5


def scale(image, label):
  image = tf.cast(image, tf.float32)
  image /= 255

  return image, label

train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
test_data = mnist_test.map(scale).batch(BATCH_SIZE)


# 간단한 예제를 위해 5개의 배치만 반환하도록 데이터셋을 자릅니다:

# In[ ]:


STEPS_PER_EPOCH = 5

train_data = train_data.take(STEPS_PER_EPOCH)
test_data = test_data.take(STEPS_PER_EPOCH)


# In[ ]:


image_batch, label_batch = next(iter(train_data))


# ### 케라스 훈련 루프 사용하기
# 
# 훈련 과정을 세부적으로 제어할 필요가 없다면 케라스의 내장 메서드인 `fit`, `evaluate`, `predict`를 사용하는 것이 좋습니다. 이 메서드들은 모델 구현(Sequential, 함수형 API, 클래스 상속)에 상관없이 일관된 훈련 인터페이스를 제공합니다.
# 
# 이 메서드들의 장점은 다음과 같습니다:
# 
# * 넘파이 배열, 파이썬 제너레이터, `tf.data.Datasets`을 사용할 수 있습니다.
# * 자동으로 규제와 활성화 손실을 적용합니다.
# * [다중 장치 훈련](distributed_training.ipynb)을 위해 `tf.distribute`을 지원합니다.
# * 임의의 호출 가능한 객체를 손실과 측정 지표로 사용할 수 있습니다.
# * `tf.keras.callbacks.TensorBoard`와 같은 콜백(callback)이나 맞춤형 콜백을 지원합니다.
# * 자동으로 텐서플로 그래프를 사용하므로 성능이 뛰어납니다.
# 
# `Dataset`을 사용하여 모델을 훈련하는 예제는 다음과 같습니다. (자세한 작동 방식은 [튜토리얼](../tutorials)을 참고하세요.)

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 맞춤형 층이 없는 모델입니다.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_data, epochs=NUM_EPOCHS)
loss, acc = model.evaluate(test_data)

print("손실 {}, 정확도 {}".format(loss, acc))


# ### 맞춤형 훈련 루프 만들기
# 
# 케라스 모델의 훈련 스텝(step)이 좋지만 그 외 다른 것을 더 제어하려면 자신만의 데이터 반복 루프를 만들고 `tf.keras.model.train_on_batch` 메서드를 사용해 보세요.
# 
# 기억할 점: 많은 것을 `tf.keras.Callback`으로 구현할 수 있습니다.
# 
# 이 메서드는 앞에서 언급한 메서드의 장점을 많이 가지고 있고 사용자가 바깥쪽 루프를 제어할 수 있습니다.
# 
# 훈련하는 동안 성능을 확인하기 위해 `tf.keras.model.test_on_batch`나 `tf.keras.Model.evaluate` 메서드를 사용할 수도 있습니다.
# 
# 노트: `train_on_batch`와 `test_on_batch`는 기본적으로 하나의 배치에 대한 손실과 측정값을 반환합니다. `reset_metrics=False`를 전달하면 누적된 측정값을 반환합니다. 이 때는 누적된 측정값을 적절하게 초기화해 주어야 합니다. `AUC`와 같은 일부 지표는 `reset_metrics=False`를 설정해야 올바르게 계산됩니다.
# 
# 앞의 모델을 계속 사용합니다:

# In[ ]:


# 맞춤형 층이 없는 모델입니다.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

metrics_names = model.metrics_names

for epoch in range(NUM_EPOCHS):
  # 누적된 측정값을 초기화합니다.
  model.reset_metrics()

  for image_batch, label_batch in train_data:
    result = model.train_on_batch(image_batch, label_batch)
    print("훈련: ",
          "{}: {:.3f}".format(metrics_names[0], result[0]),
          "{}: {:.3f}".format(metrics_names[1], result[1]))
  for image_batch, label_batch in test_data:
    result = model.test_on_batch(image_batch, label_batch,
                                 # return accumulated metrics
                                 reset_metrics=False)
  print("\n평가: ",
        "{}: {:.3f}".format(metrics_names[0], result[0]),
        "{}: {:.3f}".format(metrics_names[1], result[1]))


# <a id="custom_loops"/>
# ### 훈련 단계 커스터마이징
# 
# 자유도를 높이고 제어를 더 하려면 다음 세 단계를 사용해 자신만의 훈련 루프를 구현할 수 있습니다:
# 
# 1. 샘플 배치를 만드는 파이썬 제너레이터나 `tf.data.Dataset`을 반복합니다.
# 2. `tf.GradientTape`을 사용하여 그래디언트를 계산합니다.
# 3. `tf.keras.optimizer`를 사용하여 모델의 가중치 변수를 업데이트합니다.
# 
# 기억할 점:
# 
# * 클래스 상속으로 만든 층과 모델의 `call` 메서드에는 항상 `training` 매개변수를 포함하세요.
# * 모델을 호출할 때 `training` 매개변수를 올바르게 지정했는지 확인하세요.
# * 사용 방식에 따라 배치 데이터에서 모델이 실행될 때까지 모델 변수가 생성되지 않을 수 있습니다.
# * 모델의 규제 손실 같은 것들을 직접 관리해야 합니다.
# 
# v1에 비해 단순해진 것:
# 
# * 따로 변수를 초기화할 필요가 없습니다. 변수는 생성될 때 초기화됩니다.
# * 의존성을 수동으로 제어할 필요가 없습니다. `tf.function` 안에서도 연산은 즉시 실행 모드처럼 실행됩니다.

# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
])

optimizer = tf.keras.optimizers.Adam(0.001)
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss = tf.math.add_n(model.losses)
    pred_loss = loss_fn(labels, predictions)
    total_loss = pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

for epoch in range(NUM_EPOCHS):
  for inputs, labels in train_data:
    train_step(inputs, labels)
  print("마지막 에포크", epoch)


# ### 새로운 스타일의 측정 지표
# 
# 텐서플로 2.0에서 측정 지표는 객체입니다. 이 객체는 즉시 실행 모드와 `tf.function`에서 모두 사용할 수 있습니다. 측정 객체는 다음과 같은 메서드를 가집니다:
# 
# * `update_state()` — 새로운 측정값을 추가합니다.
# * `result()` — 누적된 측정 결과를 얻습니다.
# * `reset_states()` — 모든 측정 내용을 지웁니다.
# 
# 이 객체는 호출 가능합니다. `update_state` 메서드처럼 새로운 측정값과 함께 호출하면 상태를 업데이트하고 새로운 측정 결과를 반환합니다.
# 
# 측정 변수를 수동으로 초기화할 필요가 없습니다. 텐서플로 2.0은 자동으로 의존성을 관리하기 때문에 어떤 경우에도 신경 쓸 필요가 없습니다.
# 
# 다음은 측정 객체를 사용하여 맞춤형 훈련 루프 안에서 평균 손실을 관리하는 코드입니다.

# In[ ]:


# 측정 객체를 만듭니다.
loss_metric = tf.keras.metrics.Mean(name='train_loss')
accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

@tf.function
def train_step(inputs, labels):
  with tf.GradientTape() as tape:
    predictions = model(inputs, training=True)
    regularization_loss = tf.math.add_n(model.losses)
    pred_loss = loss_fn(labels, predictions)
    total_loss = pred_loss + regularization_loss

  gradients = tape.gradient(total_loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  # 측정값을 업데이트합니다.
  loss_metric.update_state(total_loss)
  accuracy_metric.update_state(labels, predictions)


for epoch in range(NUM_EPOCHS):
  # 측정값을 초기화합니다.
  loss_metric.reset_states()
  accuracy_metric.reset_states()

  for inputs, labels in train_data:
    train_step(inputs, labels)
  # 측정 결과를 얻습니다.
  mean_loss = loss_metric.result()
  mean_accuracy = accuracy_metric.result()

  print('에포크: ', epoch)
  print('  손실:     {:.3f}'.format(mean_loss))
  print('  정확도: {:.3f}'.format(mean_accuracy))


# ## 저장과 복원

# ### 체크포인트 호환성
# 
# 텐서플로 2.0은 [객체 기반의 체크포인트](checkpoint.ipynb)를 사용합니다.
# 
# 이전 이름 기반 스타일의 체크포인트도 여전히 복원할 수 있지만 주의가 필요합니다.
# 코드 변환 과정 때문에 변수 이름이 바뀔 수 있지만 해결 방법이 있습니다.
# 
# 가장 간단한 방법은 새로운 모델의 이름과 체크포인트에 있는 이름을 나열해 보는 것입니다:
# 
# * 여전히 모든 변수는 설정 가능한 `name` 매개변수를 가집니다.
# * 케라스 모델도 `name` 매개변수를 가집니다. 이 값은 변수 이름의 접두어로 사용됩니다.
# * `tf.name_scope` 함수를 변수 이름의 접두어를 지정하는데 사용할 수 있습니다. 이 함수는 `tf.variable_scope`와는 매우 다릅니다. 이름에만 영향을 미치며 변수를 추적하거나 재사용을 관장하지 않습니다.
# 
# 이것이 주어진 상황에 잘 맞지 않는다면 `tf.compat.v1.train.init_from_checkpoint` 함수를 시도해 보세요. 이 함수는 `assignment_map` 매개변수로 예전 이름과 새로운 이름을 매핑할 수 있습니다.
# 
# 노트: [지연 적재](checkpoint.ipynb#loading_mechanics)가 되는 객체 기반 체크포인트와는 달리 이름 기반 체크포인트는 함수가 호출될 때 모든 변수가 만들어 집니다. 일부 모델은 `build` 메서드를 호출하거나 배치 데이터에서 모델을 실행할 때까지 변수 생성을 지연합니다.

# ### saved_model 호환성
# 
# saved_model에는 심각한 호환성 문제가 없습니다.
# 
# * 텐서플로 1.x의 saved_model은 텐서플로 2.0와 호환됩니다.
# * 텐서플로 2.0의 saved_model로 저장한 모델도 연산이 지원된다면 TensorFlow 1.x에서 작동됩니다.

# ## 추정기

# ### 추정기로 훈련하기
# 
# 텐서플로 2.0은 추정기(estimator)를 지원합니다.
# 
# 추정기를 사용할 때 텐서플로 1.x의 `input_fn()`, `tf.estimator.TrainSpec`, `tf.estimator.EvalSpec`를 사용할 수 있습니다.
# 
# 다음은 `input_fn`을 사용하여 훈련과 평가를 수행하는 예입니다.

# #### input_fn과 훈련/평가 스펙 만들기

# In[ ]:


# 추정기 input_fn을 정의합니다.
def input_fn():
  datasets, info = tfds.load(name='mnist', with_info=True, as_supervised=True)
  mnist_train, mnist_test = datasets['train'], datasets['test']

  BUFFER_SIZE = 10000
  BATCH_SIZE = 64

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255

    return image, label[..., tf.newaxis]

  train_data = mnist_train.map(scale).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
  return train_data.repeat()

# 훈련과 평가 스펙을 정의합니다.
train_spec = tf.estimator.TrainSpec(input_fn=input_fn,
                                    max_steps=STEPS_PER_EPOCH * NUM_EPOCHS)
eval_spec = tf.estimator.EvalSpec(input_fn=input_fn,
                                  steps=STEPS_PER_EPOCH)


# ### 케라스 모델 정의 사용하기

# 텐서플로 2.0에서 추정기를 구성하는 방법은 조금 다릅니다.
# 
# 케라스를 사용하여 모델을 정의하는 것을 권장합니다. 그 다음 `tf.keras.model_to_estimator` 유틸리티를 사용하여 모델을 추정기로 바꾸세요. 다음 코드는 추정기를 만들고 훈련할 때 이 유틸리티를 사용하는 방법을 보여 줍니다.

# In[ ]:


def make_model():
  return tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu',
                           kernel_regularizer=tf.keras.regularizers.l2(0.02),
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dense(10, activation='softmax')
  ])


# In[ ]:


model = make_model()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(
  keras_model = model
)

tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# ### 맞춤형 `model_fn` 사용하기
# 
# 기존에 작성한 맞춤형 추정기 `model_fn`을 유지해야 한다면 이 `model_fn`을 케라스 모델로 바꿀 수 있습니다.
# 
# 그러나 호환성 때문에 맞춤형 `model_fn`은 1.x 스타일의 그래프 모드로 실행될 것입니다. 즉 즉시 실행과 의존성 자동 제어가 없다는 뜻입니다.
# 
# 맞춤형 `model_fn`에 케라스 모델을 사용하는 것은 맞춤형 훈련 루프에 사용하는 것과 비슷합니다:
# 
# * `mode` 매개변수에 기초하여 `training` 상태를 적절하게 지정하세요.
# * 옵티마이저에 모델의 `trainable_variables`를 명시적으로 전달하세요.
# 
# [맞춤형 루프](#custom_loop)와 큰 차이점은 다음과 같습니다:
# 
# * `model.losses`를 사용하는 대신 `tf.keras.Model.get_losses_for` 사용하여 손실을 추출하세요.
# * `tf.keras.Model.get_updates_for`를 사용하여 모델의 업데이트 값을 추출하세요.
# 
# 노트: "업데이트(update)"는 각 배치가 끝난 후에 모델에 적용해야 할 변화량입니다. 예를 들면 `tf.keras.layers.BatchNormalization` 층에서 평균과 분산의 이동 평균(moving average)이 있습니다.
# 
# 다음은 맞춤형 `model_fn`으로부터 추정기를 만드는 코드로 이런 개념을 잘 보여 줍니다.

# In[ ]:


def my_model_fn(features, labels, mode):
  model = make_model()

  optimizer = tf.compat.v1.train.AdamOptimizer()
  loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()

  training = (mode == tf.estimator.ModeKeys.TRAIN)
  predictions = model(features, training=training)

  reg_losses = model.get_losses_for(None) + model.get_losses_for(features)
  total_loss = loss_fn(labels, predictions) + tf.math.add_n(reg_losses)

  accuracy = tf.compat.v1.metrics.accuracy(labels=labels,
                                           predictions=tf.math.argmax(predictions, axis=1),
                                           name='acc_op')

  update_ops = model.get_updates_for(None) + model.get_updates_for(features)

  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(
        total_loss,
        var_list=model.trainable_variables,
        global_step=tf.compat.v1.train.get_or_create_global_step())

  return tf.estimator.EstimatorSpec(
    mode=mode,
    predictions=predictions,
    loss=total_loss,
    train_op=train_op, eval_metric_ops={'accuracy': accuracy})

# 추정기를 만들고 훈련합니다.
estimator = tf.estimator.Estimator(model_fn=my_model_fn)
tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)


# ## TensorShape
# 
# 이 클래스는 `tf.compat.v1.Dimension` 객체 대신에 `int` 값을 가지도록 단순화되었습니다. 따라서 `int` 값을 얻기 위해 `.value()` 메서드를 호출할 필요가 없습니다.
# 
# 여전히 개별 `tf.compat.v1.Dimension` 객체는 `tf.TensorShape.dims`로 참조할 수 있습니다.

# 다음 코드는 텐서플로 1.x와 텐서플로 2.0의 차이점을 보여줍니다.

# In[ ]:


# TensorShape 객체를 만들고 인덱스를 참조합니다.
i = 0
shape = tf.TensorShape([16, None, 256])
shape


# TF 1.x에서는 다음과 같이 사용합니다:
# 
# ```python
# value = shape[i].value
# ```
# 
# TF 2.0에서는 다음과 같이 사용합니다:

# In[ ]:


value = shape[i]
value


# TF 1.x에서는 다음과 같이 사용합니다:
# 
# ```python
# for dim in shape:
#     value = dim.value
#     print(value)
# ```
# 
# TF 2.0에서는 다음과 같이 사용합니다:

# In[ ]:


for value in shape:
  print(value)


# TF 1.x에서는 다음과 같이 사용합니다(다른 Dimension 메서드를 사용할 때도):
# 
# ```python
# dim = shape[i]
# dim.assert_is_compatible_with(other_dim)
# ```
# 
# TF 2.0에서는 다음과 같이 사용합니다:

# In[ ]:


other_dim = 16
Dimension = tf.compat.v1.Dimension

if shape.rank is None:
  dim = Dimension(None)
else:
  dim = shape.dims[i]
dim.is_compatible_with(other_dim) # 다른 Dimension 메서드도 동일


# In[ ]:


shape = tf.TensorShape(None)

if shape:
  dim = shape.dims[i]
  dim.is_compatible_with(other_dim) # 다른 Dimension 메서드도 동일


# 랭크(rank)를 알 수 있다면 `tf.TensorShape`의 불리언 값은 `True`가 됩니다. 그렇지 않으면 `False`입니다.

# In[ ]:


print(bool(tf.TensorShape([])))      # 스칼라
print(bool(tf.TensorShape([0])))     # 길이 0인 벡터
print(bool(tf.TensorShape([1])))     # 길이 1인 벡터
print(bool(tf.TensorShape([None])))  # 길이를 알 수 없는 벡터
print(bool(tf.TensorShape([1, 10, 100])))       # 3D 텐서
print(bool(tf.TensorShape([None, None, None]))) # 크기를 모르는 3D 텐서
print()
print(bool(tf.TensorShape(None)))  # 랭크를 알 수 없는 텐서


# ## 그 밖의 동작 방식 변화
# 
# 텐서플로 2.0에는 몇 가지 동작 방식의 변화가 있습니다.
# 
# ### ResourceVariables
# 
# 텐서플로 2.0은 기본적으로 `RefVariables`가 아니라 `ResourceVariables`를 만듭니다.
# 
# `ResourceVariables`는 쓰기 금지가 되어 있어서 직관적으로 일관성이 더 잘 보장됩니다.
# 
# * 이는 극단적인 경우 동작 방식에 변화를 일으킬 수 있습니다.
# * 경우에 따라서 추가적인 복사를 일으켜 메모리 사용량을 증가시킬 수 있습니다.
# * `tf.Variable` 생성자에 `use_resource=False`를 전달하여 비활성화시킬 수 있습니다.
# 
# ### 제어 흐름
# 
# 제어 흐름 연산의 구현이 단순화되었습니다. 따라서 텐서플로 2.0에서는 다른 그래프가 생성됩니다.

# ## 결론
# 
# 전체적인 과정은 다음과 같습니다:
# 
# 1. 업그레이드 스크립트를 실행하세요.
# 2. `contrib` 모듈을 삭제하세요.
# 3. 모델을 객체 지향 스타일(케라스)로 바꾸세요.
# 4. 가능한 `tf.keras`나 `tf.estimator`의 훈련과 평가 루프를 사용하세요.
# 5. 그렇지 않으면 맞춤형 루프를 사용하세요. 세션과 컬렉션은 사용하지 말아야 합니다.
# 
# 텐서플로 2.0 스타일로 코드를 바꾸려면 약간의 작업이 필요하지만 다음과 같은 장점을 얻을 수 있습니다:
# 
# * 코드 라인이 줄어 듭니다.
# * 명료하고 단순해집니다.
# * 디버깅이 쉬워집니다.
