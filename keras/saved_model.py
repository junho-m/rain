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


# # SavedModel 포맷 사용하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/saved_model">
#     <img src="https://www.tensorflow.org/images/tf_logo_32px.png" />
#     TensorFlow.org에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/saved_model.ipynb">
#     <img src="https://www.tensorflow.org/images/colab_logo_32px.png" />
#     구글 코랩(Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/saved_model.ipynb">
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

# SavedModel에는 가중치 및 연산을 포함한 완전한 텐서플로 프로그램이 포함됩니다. 기존에 설계했던 모델 코드를 실행할 필요가 없어 공유하거나 ([TFLite](https://tensorflow.org/lite), [TensorFlow.js](https://js.tensorflow.org/), [TensorFlow Serving](https://www.tensorflow.org/tfx/serving/tutorials/Serving_REST_simple), [TFHub](https://tensorflow.org/hub)와 같은 환경으로) 배포하는 데 유용합니다.
# 
# 파이썬 모델 코드를 가지고 있고 파이썬 내에서 가중치를 불러오고 싶다면, [체크포인트 훈련 가이드](./checkpoint.ipynb)를 참조하세요.
# 
# 빠른 소개를 위해 이 섹션에서는 미리 훈련된 케라스 모델을 내보내고 그 모델로 이미지 분류 요청을 처리합니다. 나머지 가이드에서는 세부 정보와 SavedModel을 만드는 다른 방법에 대해 설명합니다.

# In[ ]:


import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np


# In[ ]:


file = tf.keras.utils.get_file(
    "grace_hopper.jpg",
    "https://storage.googleapis.com/download.tensorflow.org/example_images/grace_hopper.jpg")
img = tf.keras.preprocessing.image.load_img(file, target_size=[224, 224])
plt.imshow(img)
plt.axis('off')
x = tf.keras.preprocessing.image.img_to_array(img)
x = tf.keras.applications.mobilenet.preprocess_input(
    x[tf.newaxis,...])


# 실행 예제로 그레이스 호퍼(Grace Hopper)의 이미지와 사용이 쉬운 케라스 사전 훈련 이미지 분류 모델을 사용할 것입니다. 사용자 정의 모델도 사용할 수 있는데, 자세한 것은 나중에 설명합니다.

# In[ ]:


#tf.keras.applications.vgg19.decode_predictions
labels_path = tf.keras.utils.get_file('ImageNetLabels.txt','https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt')
imagenet_labels = np.array(open(labels_path).read().splitlines())


# In[ ]:


pretrained_model = tf.keras.applications.MobileNet()
result_before_save = pretrained_model(x)
print()

decoded = imagenet_labels[np.argsort(result_before_save)[0,::-1][:5]+1]

print("저장 전 결과:\n", decoded)


# 이 이미지의 가장 가능성 있는 예측은 "군복"입니다.

# In[ ]:


tf.saved_model.save(pretrained_model, "/tmp/mobilenet/1/")


# 저장 경로의 마지막 경로 요소(여기서는 `1/`)는 모델의 버전 번호인 텐서플로 서빙(TensorFlow Serving) 컨벤션을 따릅니다 - 텐서플로 서빙과 같은 도구가 최신 모델을 구분할 수 있게 합니다.
# 
# SavedModel은 시그니처(signatures)라 불리는 이름있는 함수를 가집니다. 케라스 모델은 `serving_default` 시그니처 키를 사용하여 정방향 패스(forward pass)를 내보냅니다. [SavedModel 커맨드 라인 인터페이스](#details_of_the_savedmodel_command_line_interface)는 디스크에 저장된 SavedModel을 검사할 때 유용합니다.

# In[ ]:


get_ipython().system('saved_model_cli show --dir /tmp/mobilenet/1 --tag_set serve --signature_def serving_default')


# 파이썬에서 `tf.saved_model.load`로 SavedModel을 다시 불러오고 해군대장 호퍼(Admiral Hopper)의 이미지가 어떻게 분류되는지 볼 수 있습니다.

# In[ ]:


loaded = tf.saved_model.load("/tmp/mobilenet/1/")
print(list(loaded.signatures.keys()))  # ["serving_default"]


# 가져온 시그니처는 항상 딕셔너리를 반환합니다.

# In[ ]:


infer = loaded.signatures["serving_default"]
print(infer.structured_outputs)


# SavedModel로부터 추론을 실행하면 처음 모델과 같은 결과를 제공합니다.

# In[ ]:


labeling = infer(tf.constant(x))[pretrained_model.output_names[0]]

decoded = imagenet_labels[np.argsort(labeling)[0,::-1][:5]+1]

print("저장과 불러오기 이후의 결과:\n", decoded)


# ## 텐서플로 서빙으로 모델 배포하기
# 
# SavedModel은 파이썬에서 사용하기에 적합하지만, 일반적으로 프로덕션 환경에서는 추론을 위한 전용 서비스를 사용합니다. 이는 텐서플로 서빙을 사용한 SavedModel로 쉽게 구성할 수 있습니다.
# 
# `tensorflow_model_server`를 노트북이나 로컬 머신에 설치하는 방법을 포함한 텐서플로 서빙에 대한 자세한 내용은 [TensorFlow Serving REST 튜토리얼](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/tutorials/Serving_REST_simple.ipynb)을 참조하십시오. 간단한 예를 들면 앞서 내보낸 `mobilenet` 모델을 배포하기 위해 모델 경로를 SavedModel 디렉토리로 설정합니다:
# 
# ```bash
# nohup tensorflow_model_server \
#   --rest_api_port=8501 \
#   --model_name=mobilenet \
#   --model_base_path="/tmp/mobilenet" >server.log 2>&1
# ```
# 
#   이제 요청을 보냅니다.
# 
# ```python
# !pip install requests
# import json
# import numpy
# import requests
# data = json.dumps({"signature_name": "serving_default",
#                    "instances": x.tolist()})
# headers = {"content-type": "application/json"}
# json_response = requests.post('http://localhost:8501/v1/models/mobilenet:predict',
#                               data=data, headers=headers)
# predictions = numpy.array(json.loads(json_response.text)["predictions"])
# ```
# 
# `predictions`의 결과는 파이썬에서와 같습니다.

# ### SavedModel 포맷
# 
# SavedModel은 변수값과 상수를 포함하고 직렬화된 시그니처와 이를 실행하는 데 필요한 상태를 담은 디렉토리입니다.

# In[ ]:


get_ipython().system('ls /tmp/mobilenet/1  # assets\tsaved_model.pb\tvariables')


# `saved_model.pb` 파일은 각각 하나의 함수로 된 이름있는 시그니처 세트를 포함합니다.
# 
# SavedModel에는 다중 시그니처 세트(`saved_model_cli`의 `tag_set` 매개변수 값으로 확인된 다중 MetaGraph)를 포함할 수 있지만 이런 경우는 드뭅니다. 다중 시그니처 세트를 작성하는 API에는 [`tf.Estimator.experimental_export_all_saved_models`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#experimental_export_all_saved_models) 및 TensorFlow 1.x의 `tf.saved_model.Builder`가 포함됩니다.

# In[ ]:


get_ipython().system('saved_model_cli show --dir /tmp/mobilenet/1 --tag_set serve')


# `variables` 디렉토리에는 일반적인 훈련 체크포인트 파일이 있습니다([훈련 체크포인트 가이드](./checkpoint.ipynb) 참조).

# In[ ]:


get_ipython().system('ls /tmp/mobilenet/1/variables')


# `assets` 디렉토리에는 텐서플로 그래프(TensorFlow graph)에서 사용되는 파일들, 예를 들어 상수 테이블을 초기화하는 데 사용되는 텍스트 파일들이 있습니다. 이번 예제에서는 사용되지 않습니다.
# 
# SavedModel은 텐서플로 그래프에서 사용되지 않는 파일을 위해 `assets.extra` 디렉토리를 가질 수 있는데, 예를 들면 사용자가 SavedModel과 함께 사용할 파일입니다. 텐서플로 자체는 이 디렉토리를 사용하지 않습니다.

# ### 사용자 정의 모델 내보내기
# 
# 첫 번째 섹션에서는, `tf.saved_model.save`가 `tf.keras.Model` 객체에 대한 시그니처를 자동으로 결정했습니다. 이는 케라스의 `Model` 객체가 내보내기 위한 명시적 메서드와 입력 크기를 가지기 때문에 작동했습니다. `tf.saved_model.save`는 저수준(low-level) 모델 설계 API와도 잘 작동하지만, 모델을 텐서플로 서빙에 배포할 계획이라면 시그니처로 사용할 함수를 지정해야 합니다.

# In[ ]:


class CustomModule(tf.Module):

  def __init__(self):
    super(CustomModule, self).__init__()
    self.v = tf.Variable(1.)

  @tf.function
  def __call__(self, x):
    return x * self.v

  @tf.function(input_signature=[tf.TensorSpec([], tf.float32)])
  def mutate(self, new_v):
    self.v.assign(new_v)

module = CustomModule()


# 이 모듈은 `tf.function` 데코레이터가 적용된 두 메서드를 가지고 있습니다. 이 함수들은 SavedModel에 포함되어 있으므로 `tf.saved_model.load` 함수를 사용하여 파이썬 프로그램에 함께 로드됩니다. 하지만 명시적 선언 없이는 텐서플로 서빙과 같은 시그니처 배포 도구와 `saved_model_cli`가 접근할 수 없습니다.
# 
# `module.mutate`는 `input_signature`를 가지고 있어서 계산 그래프를 SavedModel에 저장하기 위한 정보가 이미 충분히 있습니다. `__call__`은 시그니처가 없기에 저장하기 전 이 메서드를 호출해야 합니다.

# In[ ]:


module(tf.constant(0.))
tf.saved_model.save(module, "/tmp/module_no_signatures")


# `input_signature`가 없는 함수의 경우, 저장 전에 사용된 입력의 크기는 함수가 불려진 이후에 사용될 것입니다. 스칼라값으로 `__call__`을 호출했으므로 스칼라값만 받아들일 것입니다

# In[ ]:


imported = tf.saved_model.load("/tmp/module_no_signatures")
assert 3. == imported(tf.constant(3.)).numpy()
imported.mutate(tf.constant(2.))
assert 6. == imported(tf.constant(3.)).numpy()


# 함수는 벡터와 같은 새로운 형식을 수용하지 않습니다.
# 
# ```python
# imported(tf.constant([3.]))
# ```
# 
# <pre>
# ValueError: Could not find matching function to call for canonicalized inputs ((<tf.Tensor 'args_0:0' shape=(1,) dtype=float32>,), {}). Only existing signatures are [((TensorSpec(shape=(), dtype=tf.float32, name=u'x'),), {})].
# </pre>

# `get_concrete_function`을 사용해 입력 크기를 함수 호출 없이 추가할 수 있습니다. 이 함수는 매개변수 값으로 `Tensor` 대신 입력 크기와 데이터 타입을 나타내는 `tf.TensorSpec` 객체를 받습니다. 크기가 `None`이면 모든 크기가 수용 가능합니다. 또는 각 축의 크기(axis size)를 담은 리스트일 수도 있습니다. 축 크기가 'None'이면 그 축에 대해 임의의 크기를 사용할 수 있습니다. 또한 `tf.TensorSpecs`는 이름을 가질 수 있는데, 기본값은 함수의 매개변수 키워드(여기서는 "x")입니다.

# In[ ]:


module.__call__.get_concrete_function(x=tf.TensorSpec([None], tf.float32))
tf.saved_model.save(module, "/tmp/module_no_signatures")
imported = tf.saved_model.load("/tmp/module_no_signatures")
assert [3.] == imported(tf.constant([3.])).numpy()


# `tf.keras.Model`과 `tf.Module`과 같은 객체에 포함된 함수와 변수는 가져올 때 사용할 수 있지만 많은 파이썬의 타입과 속성은 잃어버립니다. 파이썬 프로그램 자체는 SavedModel에 저장되지 않습니다.
# 
# 내보낼 함수를 시그니처로 지정하지 못했기에 시그니처는 없습니다.

# In[ ]:


get_ipython().system('saved_model_cli show --dir /tmp/module_no_signatures --tag_set serve')


# ## 내보낼 시그니처 지정하기
# 
# 어떤 함수가 시그니처라는 것을 나타내려면 저장할 때 `signatures` 매개변수를 지정합니다.

# In[ ]:


call = module.__call__.get_concrete_function(tf.TensorSpec(None, tf.float32))
tf.saved_model.save(module, "/tmp/module_with_signature", signatures=call)


# 먼저 `tf.function` 객체를 `get_concrete_function` 메서드를 사용해 `ConcreteFunction` 객체로 바꾸었습니다. 이것은 함수가 고정된 `input_signature` 없이 만들어지고 함수와 연관된 명시적인 `Tensor` 입력이 없었으므로 필수적입니다.

# In[ ]:


get_ipython().system('saved_model_cli show --dir /tmp/module_with_signature --tag_set serve --signature_def serving_default')


# In[ ]:


imported = tf.saved_model.load("/tmp/module_with_signature")
signature = imported.signatures["serving_default"]
assert [3.] == signature(x=tf.constant([3.]))["output_0"].numpy()
imported.mutate(tf.constant(2.))
assert [6.] == signature(x=tf.constant([3.]))["output_0"].numpy()
assert 2. == imported.v.numpy()


# 하나의 시그니처를 내보냈고 키는 기본값인 "serving_default"가 됩니다. 여러 시그니처를 내보내려면 딕셔너리로 전달합니다.

# In[ ]:


@tf.function(input_signature=[tf.TensorSpec([], tf.string)])
def parse_string(string_input):
  return imported(tf.strings.to_number(string_input))

signatures = {"serving_default": parse_string,
              "from_float": imported.signatures["serving_default"]}

tf.saved_model.save(imported, "/tmp/module_with_multiple_signatures", signatures)


# In[ ]:


get_ipython().system('saved_model_cli show --dir /tmp/module_with_multiple_signatures --tag_set serve')


# `saved_model_cli`는 커맨드 라인에서 SavedModel을 직접 실행할 수도 있습니다.

# In[ ]:


get_ipython().system('saved_model_cli run --dir /tmp/module_with_multiple_signatures --tag_set serve --signature_def serving_default --input_exprs="string_input=\'3.\'"')
get_ipython().system('saved_model_cli run --dir /tmp/module_with_multiple_signatures --tag_set serve --signature_def from_float --input_exprs="x=3."')


# ## 가져온 모델 미세 튜닝하기
# 
# 변수 객체가 사용 가능하므로 imported 함수를 통해 역전파할 수 있습니다.

# In[ ]:


optimizer = tf.optimizers.SGD(0.05)

def train_step():
  with tf.GradientTape() as tape:
    loss = (10. - imported(tf.constant(2.))) ** 2
  variables = tape.watched_variables()
  grads = tape.gradient(loss, variables)
  optimizer.apply_gradients(zip(grads, variables))
  return loss


# In[ ]:


for _ in range(10):
  # "v"는 5로 수렴, "loss"는 0으로 수렴
  print("loss={:.2f} v={:.2f}".format(train_step(), imported.v.numpy()))


# ## SavedModel의 제어 흐름
# 
# `tf.function`에 들어갈 수 있는 것은 모두 SavedModel에 들어갈 수 있습니다. [AutoGraph](./function.ipynb)를 사용하면 Tensor에 의존하는 조건부 논리를 파이썬 제어 흐름으로 표현할 수 있습니다.

# In[ ]:


@tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
def control_flow(x):
  if x < 0:
    tf.print("유효하지 않음!")
  else:
    tf.print(x % 3)

to_export = tf.Module()
to_export.control_flow = control_flow
tf.saved_model.save(to_export, "/tmp/control_flow")


# In[ ]:


imported = tf.saved_model.load("/tmp/control_flow")
imported.control_flow(tf.constant(-1))  # 유효하지 않음!
imported.control_flow(tf.constant(2))   # 2
imported.control_flow(tf.constant(3))   # 0


# ## 추정기(Estimator)의 SavedModel
# 
# 추정기는 [`tf.Estimator.export_saved_model`](https://www.tensorflow.org/api_docs/python/tf/estimator/Estimator#export_saved_model)을 통해 SavedModel을 내보냅니다. 자세한 내용은 [Estimator 가이드](https://www.tensorflow.org/guide/estimator)를 참조하십시오.

# In[ ]:


input_column = tf.feature_column.numeric_column("x")
estimator = tf.estimator.LinearClassifier(feature_columns=[input_column])

def input_fn():
  return tf.data.Dataset.from_tensor_slices(
    ({"x": [1., 2., 3., 4.]}, [1, 1, 0, 0])).repeat(200).shuffle(64).batch(16)
estimator.train(input_fn)

serving_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
  tf.feature_column.make_parse_example_spec([input_column]))
export_path = estimator.export_saved_model(
  "/tmp/from_estimator/", serving_input_fn)


# 이 SavedModel은 텐서플로 서빙에 배포하는 데 유용한 직렬화된 `tf.Example` 프로토콜 버퍼를 사용합니다. 그러나 `tf.saved_model.load`로 불러오고 파이썬에서 실행할 수도 있습니다.

# In[ ]:


imported = tf.saved_model.load(export_path)

def predict(x):
  example = tf.train.Example()
  example.features.feature["x"].float_list.value.extend([x])
  return imported.signatures["predict"](
    examples=tf.constant([example.SerializeToString()]))


# In[ ]:


print(predict(1.5))
print(predict(3.5))


# `tf.estimator.export.build_server_input_receiver_fn`를 사용해 `tf.train.Example`이 아닌 원시 텐서를 가지는 입력 함수를 만들 수 있습니다.

# ## C++에서 SavedModel 불러오기
# 
# SavedModel의 C++ 버전 [loader](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/cc/saved_model/loader.h)는 SessionOptions 및 RunOptions을 허용하며 경로에서 SavedModel을 불러오는 API를 제공합니다. 불러 올 그래프와 연관된 태그를 지정해야합니다. 불러온 SavedModel의 버전은 SavedModelBundle이라고 하며 MetaGraphDef와 불러온 세션을 포함합니다.
# 
# ```C++
# const string export_dir = ...
# SavedModelBundle bundle;
# ...
# LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagTrain},
#                &bundle);
# ```

# <a id=saved_model_cli/>
# 
# ## SavedModel 커맨드 라인 인터페이스 세부 사항
# 
# SavedModel 커맨드 라인 인터페이스(CLI)를 사용하여 SavedModel을 검사하고 실행할 수 있습니다.
# 예를 들어, CLI를 사용하여 모델의 `SignatureDef`를 검사할 수 있습니다.
# CLI를 사용하면 입력 Tensor 크기 및 데이터 타입이 모델과 일치하는지 신속하게 확인할 수 있습니다.
# 또한 모델을 테스트하려는 경우 다양한 형식(예를 들어, 파이썬 표현식)의 샘플 입력을
# 전달하고 출력을 가져와 CLI를 사용하여 정확성 검사를 수행할 수 있습니다.
# 
# ### SavedModel CLI 설치하기
# 
# 대체로 말하자면 다음 두 가지 방법 중 하나로 텐서플로를 설치할 수 있습니다:
# 
# *  사전에 빌드된 텐서플로 바이너리로 설치
# *  소스 코드로 텐서플로 빌드
# 
# 사전에 빌드된 텐서플로 바이너리를 통해 설치한 경우 SavedModel CLI가 이미 
# 시스템 경로 `bin\saved_model_cli`에 설치되어 있습니다.
# 
# 소스 코드에서 텐서플로를 빌드하는 경우 다음 추가 명령을 실행하여 `saved_model_cli`를 빌드해야 합니다:
# 
# ```
# $ bazel build tensorflow/python/tools:saved_model_cli
# ```
# 
# ### 명령 개요
# 
# SavedModel CLI는 SavedModel의 `MetaGraphDef`에 대해 다음 두 명령어를 지원합니다:
# 
# * SavedModel의 `MetaGraphDef`에 대한 계산을 보여주는 `show`
# * `MetaGraphDef`에 대한 계산을 실행하는 `run`
# 
# 
# ### `show` 명령어
# 
# SavedModel은 태그 세트로 식별되는 하나 이상의 `MetaGraphDef`를 포함합니다.
# 모델을 텐서플로 서빙에 배포하려면, 각 모델에 어떤 종류의 `SignatureDef`가 있는지, 그리고 입력과 출력은 무엇인지 궁금할 수 있습니다.
# `show` 명령은 SavedModel의 내용을 계층적 순서로 검사합니다. 구문은 다음과 같습니다:
# 
# ```
# usage: saved_model_cli show [-h] --dir DIR [--all]
# [--tag_set TAG_SET] [--signature_def SIGNATURE_DEF_KEY]
# ```
# 
# 예를 들어, 다음 명령은 SavedModel에서 사용 가능한 모든 `MetaGraphDef` 태그 세트를 보여줍니다:
# 
# ```
# $ saved_model_cli show --dir /tmp/saved_model_dir
# The given SavedModel contains the following tag-sets:
# serve
# serve, gpu
# ```
# 
# 다음 명령은 `MetaGraphDef`에서 사용 가능한 모든 `SignatureDef` 키를 보여줍니다:
# 
# ```
# $ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve
# The given SavedModel `MetaGraphDef` contains `SignatureDefs` with the
# following keys:
# SignatureDef key: "classify_x2_to_y3"
# SignatureDef key: "classify_x_to_y"
# SignatureDef key: "regress_x2_to_y3"
# SignatureDef key: "regress_x_to_y"
# SignatureDef key: "regress_x_to_y2"
# SignatureDef key: "serving_default"
# ```
# 
# `MetaGraphDef`가 태그 세트에 *여러 개의* 태그를 가지고 있는 경우, 모든 태그를 지정해야 하며,
# 각 태그는 쉼표로 구분해야 합니다. 예를 들어:
# 
# <pre>
# $ saved_model_cli show --dir /tmp/saved_model_dir --tag_set serve,gpu
# </pre>
# 
# 특정 `SignatureDef`에 대한 모든 입력 및 출력 텐서 정보(TensorInfo)를 표시하려면 `SignatureDef` 키를
# `signature_def` 옵션으로 전달하십시오. 이것은 나중에 계산 그래프를 실행하기 위해 입력 텐서의 텐서 키 값,
# 크기 및 데이터 타입을 알고자 할 때 매우 유용합니다. 예를 들어:
# 
# ```
# $ saved_model_cli show --dir \
# /tmp/saved_model_dir --tag_set serve --signature_def serving_default
# The given SavedModel SignatureDef contains the following input(s):
#   inputs['x'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 1)
#       name: x:0
# The given SavedModel SignatureDef contains the following output(s):
#   outputs['y'] tensor_info:
#       dtype: DT_FLOAT
#       shape: (-1, 1)
#       name: y:0
# Method name is: tensorflow/serving/predict
# ```
# 
# SavedModel에 사용 가능한 모든 정보를 표시하려면 `--all` 옵션을 사용하십시오. 예를 들어:
# 
# <pre>
# $ saved_model_cli show --dir /tmp/saved_model_dir --all
# MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:
# 
# signature_def['classify_x2_to_y3']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['inputs'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 1)
#         name: x2:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['scores'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 1)
#         name: y3:0
#   Method name is: tensorflow/serving/classify
# 
# ...
# 
# signature_def['serving_default']:
#   The given SavedModel SignatureDef contains the following input(s):
#     inputs['x'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 1)
#         name: x:0
#   The given SavedModel SignatureDef contains the following output(s):
#     outputs['y'] tensor_info:
#         dtype: DT_FLOAT
#         shape: (-1, 1)
#         name: y:0
#   Method name is: tensorflow/serving/predict
# </pre>
# 
# 
# ### `run` 명령어
# 
# `run` 명령을 호출하여 그래프 계산을 실행하고, 입력을 전달한 다음 출력을 표시(하고 선택적으로 저장)합니다.
# 구문은 다음과 같습니다:
# 
# ```
# usage: saved_model_cli run [-h] --dir DIR --tag_set TAG_SET --signature_def
#                            SIGNATURE_DEF_KEY [--inputs INPUTS]
#                            [--input_exprs INPUT_EXPRS]
#                            [--input_examples INPUT_EXAMPLES] [--outdir OUTDIR]
#                            [--overwrite] [--tf_debug]
# ```
# 
# `run` 명령은 입력을 모델에 전달하는 다음 세 가지 방법을 제공합니다:
# 
# * `--inputs` 옵션을 사용하여 넘파이(numpy) ndarray를 파일에 전달할 수 있습니다.
# * `--input_exprs` 옵션을 사용하여 파이썬 표현식을 전달할 수 있습니다.
# * `--input_examples` 옵션을 사용하여 `tf.train.Example`을 전달할 수 있습니다.
# 
# #### `--inputs`
# 
# 입력 데이터를 파일에 전달하려면, 다음과 같은 일반적인 형식을 가지는 `--inputs` 옵션을 지정합니다:
# 
# ```bsh
# --inputs <INPUTS>
# ```
# 
# 여기서 *INPUTS*는 다음 형식 중 하나입니다:
# 
# *  `<input_key>=<filename>`
# *  `<input_key>=<filename>[<variable_name>]`
# 
# 여러 개의 *INPUTS*를 전달할 수 있습니다. 여러 입력을 전달하는 경우 세미콜론을 사용하여 각 *INPUTS*를 구분하십시오.
# 
# `saved_model_cli`는 `numpy.load`를 사용하여 *filename*을 불러옵니다.
# *filename*은 다음 형식 중 하나일 수 있습니다:
# 
# *  `.npy`
# *  `.npz`
# *  피클(pickle) 포맷
# 
# `.npy` 파일은 항상 넘파이 ndarray를 포함합니다. 그러므로 `.npy` 파일에서 불러올 때,
# 배열 내용이 지정된 입력 텐서에 직접 할당될 것입니다. 해당 `.npy` 파일과 함께 *variable_name*을 지정하면
# *variable_name*이 무시되고 경고가 발생합니다.
# 
# `.npz`(zip) 파일에서 불러올 때, 입력 텐서 키로 불러올 zip 파일 내의 변수를 *variable_name*으로
# 선택적으로 지정할 수 있습니다. *variable_name*을 지정하지 않으면 SavedModel CLI는 zip 파일에 하나의 파일만
# 포함되어 있는지 확인하고 지정된 입력 텐서 키로 불러옵니다.
# 
# 피클 파일에서 불러올 때, 대괄호 안에 `variable_name`이 지정되지 않았다면, 피클 파일 안에 있는
# 어떤 것이라도 지정된 입력 텐서 키로 전달될 것입니다. 그렇지 않으면, SavedModel CLI는 피클 파일에
# 딕셔너리가 저장되어 있다고 가정하고 *variable_name*에 해당하는 값이 사용됩니다.
# 
# #### `--input_exprs`
# 
# 파이썬 표현식을 통해 입력을 전달하려면 `--input_exprs` 옵션을 지정하십시오. 이는 데이터 파일이 없어도
# 모델의 `SignatureDef`의 크기 및 데이터 타입과 일치하는 간단한 입력으로 모델의 정확성 검사를 하려는 경우
# 유용할 수 있습니다. 예를 들어:
# 
# ```bsh
# `<input_key>=[[1],[2],[3]]`
# ```
# 
# 파이썬 표현식 외에도 넘파이 함수를 전달할 수 있습니다. 예를 들어:
# 
# ```bsh
# `<input_key>=np.ones((32,32,3))`
# ```
# 
# (`numpy` 모듈은 `np`로 이미 사용 가능하다고 가정합니다.)
# 
# 
# #### `--input_examples`
# 
# `tf.train.Example`을 입력으로 전달하려면 `--input_examples` 옵션을 지정하십시오. 입력 키마다 딕셔너리의
# 리스트를 받습니다. 각 딕셔너리는 `tf.train.Example`의 인스턴스입니다. 딕셔너리 키는 기능이며 값은 각 기능의
# 값 리스트입니다. 예를 들어:
# 
# ```bsh
# `<input_key>=[{"age":[22,24],"education":["BS","MS"]}]`
# ```
# 
# #### 출력 저장
# 
# 기본적으로, SavedModel CLI는 출력을 stdout에 기록합니다. `--outdir` 옵션으로 디렉토리를 전달하면,
# 지정된 디렉토리 안에 출력 텐서 키의 이름을 따라 .npy 파일로 출력이 저장됩니다.
# 
# 기존 출력 파일을 덮어 쓰려면 `--overwrite`를 사용하십시오.
