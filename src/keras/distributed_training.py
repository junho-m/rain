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


# # 텐서플로로 분산 훈련하기

# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/distributed_training"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />TensorFlow.org에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/distributed_training.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/distributed_training.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />깃허브(GitHub) 소스 보기</a>
#   </td>
# </table>

# Note: 이 문서는 텐서플로 커뮤니티에서 번역했습니다. 커뮤니티 번역 활동의 특성상 정확한 번역과 최신 내용을 반영하기 위해 노력함에도
# 불구하고 [공식 영문 문서](https://www.tensorflow.org/?hl=en)의 내용과 일치하지 않을 수 있습니다.
# 이 번역에 개선할 부분이 있다면
# [tensorflow/docs-l10n](https://github.com/tensorflow/docs-l10n/) 깃헙 저장소로 풀 리퀘스트를 보내주시기 바랍니다.
# 문서 번역이나 리뷰에 참여하려면
# [docs-ko@tensorflow.org](https://groups.google.com/a/tensorflow.org/forum/#!forum/docs-ko)로
# 메일을 보내주시기 바랍니다.

# ## 개요
# 
# `tf.distribute.Strategy`는 훈련을 여러 GPU 또는 여러 장비, 여러 TPU로 나누어 처리하기 위한 텐서플로 API입니다. 이 API를 사용하면 기존의 모델이나 훈련 코드를 조금만 고쳐서 분산처리를 할 수 있습니다.
# 
# `tf.distribute.Strategy`는 다음을 주요 목표로 설계하였습니다.
# 
# * 사용하기 쉽고, 연구원, 기계 학습 엔지니어 등 여러 사용자 층을 지원할 것.
# * 그대로 적용하기만 하면 좋은 성능을 보일 것.
# * 전략들을 쉽게 갈아 끼울 수 있을 것.
# 
# `tf.distribute.Strategy`는 텐서플로의 고수준 API인 [tf.keras](https://www.tensorflow.org/guide/keras) 및 [tf.estimator](https://www.tensorflow.org/guide/estimator)와 함께 사용할 수 있습니다. 코드 한두 줄만 추가하면 됩니다. 사용자 정의 훈련 루프(그리고 텐서플로를 사용한 모든 계산 작업)에 함께 사용할 수 있는 API도 제공합니다.
# 텐서플로 2.0에서는 사용자가 프로그램을 즉시 실행(eager execution)할 수도 있고, [`tf.function`](../tutorials/eager/tf_function.ipynb)을 사용하여 그래프에서 실행할 수도 있습니다. `tf.distribute.Strategy`는 두 가지 실행 방식을 모두 지원하려고 합니다. 이 가이드에서는 대부분의 경우 훈련에 대하여 이야기하겠지만, 이 API 자체는 여러 환경에서 평가나 예측을 분산 처리하기 위하여 사용할 수도 있다는 점을 참고하십시오.
# 
# 잠시 후 보시겠지만 코드를 약간만 바꾸면 `tf.distribute.Strategy`를 사용할 수 있습니다. 변수, 층, 모델, 옵티마이저, 지표, 서머리(summary), 체크포인트 등 텐서플로를 구성하고 있는 기반 요소들을 전략(Strategy)을 이해하고 처리할 수 있도록 수정했기 때문입니다. 
# 
# 이 가이드에서는 다양한 형식의 전략에 대해서, 그리고 여러 가지 상황에서 이들을 어떻게 사용해야 하는지 알아보겠습니다.

# In[ ]:


# 텐서플로 패키지 가져오기
get_ipython().system('pip install tensorflow-gpu==2.0.0-rc1')
import tensorflow as tf


# ## 전략의 종류
# `tf.distribute.Strategy`는 서로 다른 다양한 사용 형태를 아우르려고 합니다. 몇 가지 조합은 현재 지원하지만, 추후에 추가될 전략들도 있습니다. 이들 중 몇 가지를 살펴보겠습니다.
# 
# * 동기 훈련 대 비동기 훈련: 분산 훈련을 할 때 데이터를 병렬로 처리하는 방법은 크게 두 가지가 있습니다. 동기 훈련을 할 때는 모든 워커(worker)가 입력 데이터를 나누어 갖고 동시에 훈련합니다. 그리고 각 단계마다 그래디언트(gradient)를 모읍니다. 비동기 훈련에서는 모든 워커가 독립적으로 입력 데이터를 사용해 훈련하고 각각 비동기적으로 변수들을 갱신합니다. 일반적으로 동기 훈련은 올 리듀스(all-reduce)방식으로 구현하고, 비동기 훈련은 파라미터 서버 구조를 사용합니다.
# * 하드웨어 플랫폼: 한 장비에 있는 다중 GPU로 나누어 훈련할 수도 있고, 네트워크로 연결된 (GPU가 없거나 여러 개의 GPU를 가진) 여러 장비로 나누어서, 또 혹은 클라우드 TPU에서 훈련할 수도 있습니다.
# 
# 이런 사용 형태들을 위하여, 현재 5가지 전략을 사용할 수 있습니다. 이후 내용에서 현재 TF 2.0 베타에서 상황마다 어떤 전략을 지원하는지 이야기하겠습니다. 일단 간단한 개요는 다음과 같습니다.
# 
# | 훈련 API          	| MirroredStrategy  	| TPUStrategy         	| MultiWorkerMirroredStrategy     	| CentralStorageStrategy          	| ParameterServerStrategy  	|
# |:-----------------------	|:-------------------	|:---------------------	|:---------------------------------	|:---------------------------------	|:--------------------------	|
# | **Keras API**             	| 지원	| 2.0 RC 지원 예정	| 실험 기능으로 지원	| 실험 기능으로 지원	| 2.0 이후 지원 예정	|
# | **사용자 정의 훈련 루프**  	| 실험 기능으로 지원	| 실험 기능으로 지원  	| 2.0 이후 지원 예정	| 2.0 RC 지원 예정	| 아직 미지원	|
# | **Estimator API**         	| 제한적으로 지원	| 제한적으로 지원	| 제한적으로 지원	| 제한적으로 지원	| 제한적으로 지원	|

# ### MirroredStrategy
# `tf.distribute.MirroredStrategy`는 장비 하나에서 다중 GPU를 이용한 동기 분산 훈련을 지원합니다. 각각의 GPU 장치마다 복제본이 만들어집니다. 모델의 모든 변수가 복제본마다 미러링 됩니다. 이 미러링된 변수들은 하나의 가상의 변수에 대응되는데, 이를 `MirroredVariable`라고 합니다. 이 변수들은 동일한 변경사항이 함께 적용되므로 모두 같은 값을 유지합니다.
# 
# 여러 장치에 변수의 변경사항을 전달하기 위하여 효율적인 올 리듀스 알고리즘을 사용합니다. 올 리듀스 알고리즘은 모든 장치에 걸쳐 텐서를 모은 다음, 그 합을 구하여 다시 각 장비에 제공합니다. 이 통합된 알고리즘은 매우 효율적이어서 동기화의 부담을 많이 덜어낼 수 있습니다. 장치 간에 사용 가능한 통신 방법에 따라 다양한 올 리듀스 알고리즘과 구현이 있습니다. 기본값으로는 NVIDIA NCCL을 올 리듀스 구현으로 사용합니다. 또한 제공되는 다른 몇 가지 방법 중에 선택하거나, 직접 만들 수도 있습니다.
# 
# `MirroredStrategy`를 만드는 가장 쉬운 방법은 다음과 같습니다.

# In[ ]:


mirrored_strategy = tf.distribute.MirroredStrategy()


# `MirroredStrategy` 인스턴스가 생겼습니다. 텐서플로가 인식한 모든 GPU를 사용하고, 장치 간 통신에는 NCCL을 사용할 것입니다.
# 
# 장비의 GPU 중 일부만 사용하고 싶다면, 다음과 같이 하면 됩니다.

# In[ ]:


mirrored_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


# 장치 간 통신 방법을 바꾸고 싶다면, `cross_device_ops` 인자에 `tf.distribute.CrossDeviceOps` 타입의 인스턴스를 넘기면 됩니다. 현재 기본값인 `tf.distribute.NcclAllReduce` 이외에 `tf.distribute.HierarchicalCopyAllReduce`와 `tf.distribute.ReductionToOneDevice` 두 가지 추가 옵션을 제공합니다.

# In[ ]:


mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())


# ### CentralStorageStrategy
# `tf.distribute.experimental.CentralStorageStrategy`도 동기 훈련을 합니다. 하지만 변수를 미러링하지 않고, CPU에서 관리합니다. 작업은 모든 로컬 GPU들로 복제됩니다. 단, 만약 GPU가 하나밖에 없다면 모든 변수와 작업이 그 GPU에 배치됩니다.
# 
# 다음과 같이 `CentralStorageStrategy` 인스턴스를 만드십시오.

# In[ ]:


central_storage_strategy = tf.distribute.experimental.CentralStorageStrategy()


# `CentralStorageStrategy` 인스턴스가 만들어졌습니다. 인식한 모든 GPU와 CPU를 사용합니다. 각 복제본의 변수 변경사항은 모두 수집된 후 변수에 적용됩니다.

# Note: 이 전략은 아직 개선 중이고 더 많은 경우에 쓸 수 있도록 만들고 있기 때문에, [`실험 기능`](https://www.tensorflow.org/guide/versions#what_is_not_covered)으로 지원됩니다. 따라서 다음에 API가 바뀔 수 있음에 유념하십시오.

# ### MultiWorkerMirroredStrategy
# 
# `tf.distribute.experimental.MultiWorkerMirroredStrategy`은 `MirroredStrategy`와 매우 비슷합니다. 다중 워커를 이용하여 동기 분산 훈련을 합니다. 각 워커는 여러 개의 GPU를 사용할 수 있습니다. `MirroredStrategy`처럼 모델에 있는 모든 변수의 복사본을 모든 워커의 각 장치에 만듭니다.
# 
# 다중 워커(multi-worker)들 사이에서는 올 리듀스(all-reduce) 통신 방법으로 [CollectiveOps](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/collective_ops.py)를 사용하여 변수들을 같은 값으로 유지합니다. 수집 연산(collective op)은 텐서플로 그래프에 속하는 연산 중 하나입니다. 이 연산은 하드웨어나 네트워크 구성, 텐서 크기에 따라 텐서플로 런타임이 지원하는 올 리듀스 알고리즘을 자동으로 선택합니다.
# 
# 여기에 추가 성능 최적화도 구현하고 있습니다. 예를 들어 작은 텐서들의 여러 올 리듀스 작업을 큰 텐서들의 더 적은 올 리듀스 작업으로 바꾸는 정적 최적화 기능이 있습니다. 뿐만아니라 플러그인 구조를 갖도록 설계하였습니다. 따라서 추후에는 사용자가 자신의 하드웨어에 더 최적화된 알고리즘을 사용할 수도 있을 것입니다. 참고로 이 수집 연산은 올 리듀스 외에 브로드캐스트(broadcast)나 전체 수집(all-gather)도 구현하고 있습니다.
# 
# `MultiWorkerMirroredStrategy`를 만드는 가장 쉬운 방법은 다음과 같습니다.

# In[ ]:


multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()


# `MultiWorkerMirroredStrategy`에 사용할 수 있는 수집 연산 구현은 현재 두 가지입니다. `CollectiveCommunication.RING`는 gRPC를 사용한 링 네트워크 기반의 수집 연산입니다. `CollectiveCommunication.NCCL`는 [Nvidia의 NCCL](https://developer.nvidia.com/nccl)을 사용하여 수집 연산을 구현한 것입니다. `CollectiveCommunication.AUTO`로 설정하면 런타임이 알아서 구현을 고릅니다. 최적의 수집 연산 구현은 GPU의 수와 종류, 클러스터의 네트워크 연결 등에 따라 다를 수 있습니다. 예를 들어 다음과 같이 지정할 수 있습니다.

# In[ ]:


multiworker_strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.NCCL)


# 다중 GPU를 사용하는 것과 비교해서 다중 워커를 사용하는 것의 가장 큰 차이점은 다중 워커에 대한 설정 부분입니다. 클러스터를 구성하는 각 워커에 "TF_CONFIG" 환경변수를 사용하여 클러스터 설정을 하는 것이 텐서플로의 표준적인 방법입니다. [아래쪽 "TF_CONFIG"](#TF_CONFIG) 항목에서 어떻게 하는지 자세히 살펴보겠습니다.

# Note: 이 전략은 아직 개선 중이고 더 많은 경우에 쓸 수 있도록 만들고 있기 때문에, [`실험 기능`](https://www.tensorflow.org/guide/versions#what_is_not_covered)으로 지원됩니다. 따라서 나중에 API가 바뀔 수 있음에 유념하십시오.

# ### TPUStrategy
# `tf.distribute.experimental.TPUStrategy`는 텐서플로 훈련을 텐서처리장치(Tensor Processing Unit, TPU)에서 수행하는 전략입니다. TPU는 구글의 특별한 주문형 반도체(ASIC)로서, 기계 학습 작업을 극적으로 가속하기 위하여 설계되었습니다. TPU는 구글 코랩, [Tensorflow Research Cloud](https://www.tensorflow.org/tfrc), [Google Compute Engine](https://cloud.google.com/tpu)에서 사용할 수 있습니다.
# 
# 분산 훈련 구조의 측면에서, TPUStrategy는 `MirroredStrategy`와 동일합니다. 동기 분산 훈련 방식을 사용합니다. TPU는 자체적으로 여러 TPU 코어들에 걸친 올 리듀스 및 기타 수집 연산을 효율적으로 구현하고 있습니다. 이 구현이 `TPUStrategy`에 사용됩니다.
# 
# `TPUStrategy`를 사용하는 방법은 다음과 같습니다.
# 
# Note: 코랩에서 이 코드를 사용하려면, 코랩 런타임으로 TPU를 선택해야 합니다. TPUStrategy를 사용하는 방법에 대한 튜토리얼을 곧 추가하겠습니다.
# 
# ```
# cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
#     tpu=tpu_address)
# tf.config.experimental_connect_to_host(cluster_resolver.master())
# tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
# tpu_strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)
# ```

# `TPUClusterResolver` 인스턴스는 TPU를 찾도록 도와줍니다. 코랩에서는 아무런 인자를 주지 않아도 됩니다. 클라우드 TPU에서 사용하려면, TPU 자원의 이름을 `tpu` 매개변수에 지정해야 합니다. 또한 TPU는 계산하기 전 초기화(initialize)가 필요합니다. 초기화 중 TPU 메모리가 지워져서 모든 상태 정보가 사라지므로, 프로그램 시작시에 명시적으로 TPU 시스템을 초기화(initialize)해 주어야 합니다.

# Note: 이 전략은 아직 개선 중이고 더 많은 경우에 쓸 수 있도록 만들고 있기 때문에, [`실험 기능`](https://www.tensorflow.org/guide/versions#what_is_not_covered)으로 지원됩니다. 따라서 나중에 API가 바뀔 수 있음에 유념하십시오.

# ### ParameterServerStrategy
# `tf.distribute.experimental.ParameterServerStrategy`은 여러 장비에서 훈련할 때 파라미터 서버를 사용합니다. 이 전략을 사용하면 몇 대의 장비는 워커 역할을 하고, 몇 대는 파라미터 서버 역할을 하게 됩니다. 모델의 각 변수는 한 파라미터 서버에 할당됩니다. 계산 작업은 모든 워커의 GPU들에 복사됩니다.
# 
# 코드만 놓고 보았을 때는 다른 전략들과 비슷합니다.
# ```
# ps_strategy = tf.distribute.experimental.ParameterServerStrategy()
# ```

# 다중 워커 환경에서 훈련하려면, 클러스터에 속한 파라미터 서버와 워커를 "TF_CONFIG" 환경변수를 이용하여 설정해야 합니다. 자세한 내용은 [아래쪽 "TF_CONFIG"](#TF_CONFIG)에서 설명하겠습니다.

# 여기까지 여러 가지 전략들이 어떻게 다르고, 어떻게 사용하는지 살펴보았습니다. 이어지는 절들에서는 훈련을 분산시키기 위하여 이들을 어떻게 사용해야 하는지 살펴보겠습니다. 이 문서에서는 간단한 코드 조각만 보여드리겠지만, 처음부터 끝까지 전체 코드를 실행할 수 있는 더 긴 튜토리얼의 링크도 함께 안내해드리겠습니다.

# ## 케라스와 함께 `tf.distribute.Strategy` 사용하기
# `tf.distribute.Strategy`는 텐서플로의 [케라스 API 명세](https://keras.io) 구현인 `tf.keras`와 함께 사용할 수 있습니다. `tf.keras`는 모델을 만들고 훈련하는 고수준 API입니다. 분산 전략을 `tf.keras` 백엔드와 함께 쓸 수 있으므로, 케라스 사용자들도 케라스 훈련 프레임워크로 작성한 훈련 작업을 쉽게 분산 처리할 수 있게 되었습니다. 훈련 프로그램에서 고쳐야하는 부분은 거의 없습니다. (1) 적절한 `tf.distribute.Strategy` 인스턴스를 만든 다음 (2) 
# 케라스 모델의 생성과 컴파일을 `strategy.scope` 안으로 옮겨주기만 하면 됩니다. `Sequential` , 함수형 API, 클래스 상속 등 모든 방식으로 만든 케라스 모델을 다 지원합니다.
# 
# 다음은 한 개의 밀집 층(dense layer)을 가진 매우 간단한 케라스 모델에 분산 전략을 사용하는 코드의 일부입니다.

# In[ ]:


mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  model.compile(loss='mse', optimizer='sgd')


# 위 예에서는 `MirroredStrategy`를 사용했기 때문에, 하나의 장비가 다중 GPU를 가진 경우에 사용할 수 있습니다. `strategy.scope()`로 분산 처리할 부분을 코드에 지정할 수 있습니다. 이 범위(scope) 안에서 모델을 만들면, 일반적인 변수가 아니라 미러링된 변수가 만들어집니다. 이 범위 안에서 컴파일을 한다는 것은 작성자가 이 전략을 사용하여 모델을 훈련하려고 한다는 의미입니다. 이렇게 구성하고 나서, 일반적으로 실행하는 것처럼 모델의 `fit` 함수를 호출합니다.
# `MirroredStrategy`가 모델의 훈련을 사용 가능한 GPU들로 복제하고, 그래디언트들을 수집하는 것 등을 알아서 처리합니다.

# In[ ]:


dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
model.evaluate(dataset)


# 위에서는 훈련과 평가 입력을 위해 `tf.data.Dataset`을 사용했습니다. 넘파이(numpy) 배열도 사용할 수 있습니다.

# In[ ]:


import numpy as np
inputs, targets = np.ones((100, 1)), np.ones((100, 1))
model.fit(inputs, targets, epochs=2, batch_size=10)


# 데이터셋이나 넘파이를 사용하는 두 경우 모두 입력 배치가 동일한 크기로 나누어져서 여러 개로 복제된 작업에 전달됩니다. 예를 들어, `MirroredStrategy`를 2개의 GPU에서 사용한다면, 크기가 10개인 배치(batch)가 두 개의 GPU로 배분됩니다. 즉, 각 GPU는 한 단계마다 5개의 입력을 받게 됩니다. 따라서 GPU가 추가될수록 각 에포크(epoch) 당 훈련 시간은 줄어들게 됩니다. 일반적으로는 가속기를 더 추가할 때마다 배치 사이즈도 더 키웁니다. 추가한 컴퓨팅 자원을 더 효과적으로 사용하기 위해서입니다. 모델에 따라서는 학습률(learning rate)을 재조정해야 할 수도 있을 것입니다. 복제본의 수는 `strategy.num_replicas_in_sync`로 얻을 수 있습니다.

# In[ ]:


# 복제본의 수로 전체 배치 크기를 계산.
BATCH_SIZE_PER_REPLICA = 5
global_batch_size = (BATCH_SIZE_PER_REPLICA *
                     mirrored_strategy.num_replicas_in_sync)
dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100)
dataset = dataset.batch(global_batch_size)

LEARNING_RATES_BY_BATCH_SIZE = {5: 0.1, 10: 0.15}
learning_rate = LEARNING_RATES_BY_BATCH_SIZE[global_batch_size]


# ### 현재 어떤 것이 지원됩니까?
# 
# TF 2.0 베타 버전에서는 케라스와 함께 `MirroredStrategy`와 `CentralStorageStrategy`, `MultiWorkerMirroredStrategy`를 사용하여 훈련할 수 있습니다. `CentralStorageStrategy`와 `MultiWorkerMirroredStrategy`는 아직 실험 기능이므로 추후 바뀔 수 있습니다.
# 다른 전략도 조만간 지원될 것입니다. API와 사용 방법은 위에 설명한 것과 동일할 것입니다.
# 
# | 훈련 API 	| MirroredStrategy  	| TPUStrategy         	| MultiWorkerMirroredStrategy     	| CentralStorageStrategy          	| ParameterServerStrategy 	|
# |----------------	|---------------------	|-----------------------	|-----------------------------------	|-----------------------------------	|---------------------------	|
# | Keras API   	| 지원 	| 2.0 RC 지원 예정 	| 실험 기능으로 지원	| 실험 기능으로 지원 	| 2.0 RC 지원 예정	|
# 
# ### 예제와 튜토리얼
# 
# 위에서 설명한 케라스 분산 훈련 방법에 대한 튜토리얼과 예제들의 목록입니다.
# 
# 1. `MirroredStrategy`를 사용한 [MNIST](../tutorials/distribute/keras.ipynb) 훈련 튜토리얼.
# 2. ImageNet 데이터와 `MirroredStrategy`를 사용한 공식 [ResNet50](https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet_imagenet_main.py) 훈련.
# 3. 클라우드 TPU에서 ImageNet 데이터와 `TPUStrategy`를 사용한 [ResNet50](https://github.com/tensorflow/tpu/blob/master/models/experimental/resnet50_keras/resnet50.py) 훈련. 이 예제는 현재 텐서플로 1.x 버전에서만 동작합니다.
# 4. `MultiWorkerMirroredStrategy`를 사용한 [MNIST](../tutorials/distribute/multi_worker_with_keras.ipynb) 훈련 튜토리얼.
# 5. `MirroredStrategy`를 사용한 [NCF](https://github.com/tensorflow/models/blob/master/official/recommendation/ncf_keras_main.py) 훈련.
# 6. `MirroredStrategy`를 사용한 [Transformer](https://github.com/tensorflow/models/blob/master/official/nlp/transformer/transformer_main.py) 훈련.

# ## 사용자 정의 훈련 루프와 함께 `tf.distribute.Strategy` 사용하기
# 지금까지 살펴본 것처럼 고수준 API와 함께 `tf.distribute.Strategy`를 사용하려면 코드 몇 줄만 바꾸면 되었습니다. 조금만 더 노력을 들이면 이런 프레임워크를 사용하지 않는 사용자도 `tf.distribute.Strategy`를 사용할 수 있습니다.
# 
# 텐서플로는 다양한 용도로 사용됩니다. 연구자들 같은 일부 사용자들은 더 높은 자유도와 훈련 루프에 대한 제어를 원합니다. 이 때문에 추정기나 케라스 같은 고수준 API를 사용하기 힘든 경우가 있습니다. 예를 들어, GAN을 사용하는데 매번 생성자(generator)와 판별자(discriminator) 단계의 수를 바꾸고 싶을 수 있습니다. 비슷하게, 고수준 API는 강화 학습(Reinforcement learning)에는 그다지 적절하지 않습니다. 그래서 이런 사용자들은 보통 자신만의 훈련 루프를 작성하게 됩니다.
# 
# 이 사용자들을 위하여, `tf.distribute.Strategy` 클래스들은 일련의 주요 메서드들을 제공합니다. 이 메서드들을 사용하려면 처음에는 코드를 이리저리 조금 옮겨야 할 수 있겠지만, 한번 작업해 놓으면 전략 인스턴스만 바꿔서 GPU, TPU, 여러 장비로 쉽게 바꿔가며 훈련을 할 수 있습니다.
# 
# 앞에서 살펴본 케라스 모델을 사용한 훈련 예제를 통하여 사용하는 모습을 간단하게 살펴보겠습니다.

# 먼저, 전략의 범위(scope) 안에서 모델과 옵티마이저를 만듭니다. 이는 모델이나 옵티마이저로 만들어진 변수가 미러링 되도록 만듭니다.

# In[ ]:


with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
  optimizer = tf.keras.optimizers.SGD()


# 다음으로는 입력 데이터셋을 만든 다음, `tf.distribute.Strategy.experimental_distribute_dataset` 메서드를 호출하여 전략에 맞게 데이터셋을 분배합니다.

# In[ ]:


with mirrored_strategy.scope():
  dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(1000).batch(
      global_batch_size)
  dist_dataset = mirrored_strategy.experimental_distribute_dataset(dataset)


# 그리고 나서는 한 단계의 훈련을 정의합니다. 그래디언트를 계산하기 위해 `tf.GradientTape`를 사용합니다. 이 그래디언트를 적용하여 우리 모델의 변수를 갱신하기 위해서는 옵티마이저를 사용합니다. 분산 훈련을 위하여 이 훈련 작업을 `step_fn` 함수 안에 구현합니다. 그리고 `step_fn`을 앞에서 만든 `dist_dataset`에서 얻은 입력 데이터와 함께 `tf.distrbute.Strategy.experimental_run_v2`메서드로 전달합니다.

# In[ ]:


@tf.function
def train_step(dist_inputs):
  def step_fn(inputs):
    features, labels = inputs

    with tf.GradientTape() as tape:
      logits = model(features)
      cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
          logits=logits, labels=labels)
      loss = tf.reduce_sum(cross_entropy) * (1.0 / global_batch_size)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(list(zip(grads, model.trainable_variables)))
    return cross_entropy

  per_example_losses = mirrored_strategy.experimental_run_v2(
      step_fn, args=(dist_inputs,))
  mean_loss = mirrored_strategy.reduce(
      tf.distribute.ReduceOp.MEAN, per_example_losses, axis=0)
  return mean_loss


# 위 코드에서 몇 가지 더 짚어볼 점이 있습니다.
# 
# 1. 손실(loss)을 계산하기 위하여 `tf.nn.softmax_cross_entropy_with_logits`를 사용하였습니다. 그리고 손실의 합을 전체 배치 크기로 나누는 부분이 중요합니다. 이는 모든 복제된 훈련이 동시에 이루어지고 있고, 각 단계에 훈련이 이루어지는 입력의 수는 전체 배치 크기와 같기 때문입니다. 따라서 손실 값은 각 복제된 작업 내의 배치 크기가 아니라 전체 배치 크기로 나누어야 맞습니다.
# 2. `tf.distribute.Strategy.experimental_run_v2`에서 반환된 결과를 모으기 위하여 `tf.distribute.Strategy.reduce` API를 사용하였습니다. `tf.distribute.Strategy.experimental_run_v2`는 전략의 각 복제본에서 얻은 결과를 반환합니다. 그리고 이 결과를 사용하는 방법은 여러 가지가 있습니다. 종합한 결과를 얻기 위하여 `reduce` 함수를 사용할 수 있습니다. `tf.distribute.Strategy.experimental_local_results` 메서드로 각 복제본에서 얻은 결과의 값들 목록을 얻을 수도 있습니다.
# 3. 분산 전략 범위 안에서 `apply_gradients` 메서드가 호출되면, 평소와는 동작이 다릅니다. 구체적으로는 동기화된 훈련 중 병렬화된 각 작업에서 그래디언트를 적용하기 전에, 모든 복제본의 그래디언트를 더해집니다.

# 훈련 단계를 정의했으므로, 마지막으로는 `dist_dataset`에 대하여 훈련을 반복합니다.

# In[ ]:


with mirrored_strategy.scope():
  for inputs in dist_dataset:
    print(train_step(inputs))


# 위 예에서는 `dist_dataset`을 차례대로 처리하며 훈련 입력 데이터를 얻었습니다. `tf.distribute.Strategy.make_experimental_numpy_dataset`를 사용하면 넘파이 입력도 쓸 수 있습니다. `tf.distribute.Strategy.experimental_distribute_dataset` 함수를 호출하기 전에 이 API로 데이터셋을 만들면 됩니다.
# 
# 데이터를 차례대로 처리하는 또 다른 방법은 명시적으로 반복자(iterator)를 사용하는 것입니다. 전체 데이터를 모두 사용하지 않고, 정해진 횟수만큼만 훈련을 하고 싶을 때 유용합니다. 반복자를 만들고 명시적으로 `next`를 호출하여 다음 입력 데이터를 얻도록 하면 됩니다. 위 루프 코드를 바꿔보면 다음과 같습니다. 

# In[ ]:


with mirrored_strategy.scope():
  iterator = iter(dist_dataset)
  for _ in range(10):
    print(train_step(next(iterator)))


# `tf.distribute.Strategy` API를 사용하여 사용자 정의 훈련 루프를 분산 처리 하는 가장 단순한 경우를 살펴보았습니다. 현재 API를 개선하는 과정 중에 있습니다. 이 API를 사용하려면 사용자 쪽에서 꽤 많은 작업을 해야 하므로, 나중에 별도의 더 자세한 가이드로 설명하도록 하겠습니다.

# ### 현재 어떤 것이 지원됩니까?
# TF 2.0 베타 버전에서는 사용자 정의 훈련 루프와 함께 위에서 설명한 `MirroredStrategy`, 그리고 `TPUStrategy`를 사용할 수 있습니다. 또한 `MultiWorkerMirorredStrategy`도 추후 지원될 것입니다.
# 
# | 훈련 API          	| MirroredStrategy  	| TPUStrategy       	| MultiWorkerMirroredStrategy 	| CentralStorageStrategy 	| ParameterServerStrategy 	|
# |:-----------------------	|:-------------------	|:-------------------	|:-----------------------------	|:------------------------	|:-------------------------	|
# | 사용자 정의 훈련 루프 	| 지원 	| 지원 	| 2.0 RC 지원 예정         	| 2.0 RC 지원 예정    	| 아직 미지원      	|
# 
# ### 예제와 튜토리얼
# 사용자 정의 훈련 루프와 함께 분산 전략을 사용하는 예제들입니다.
# 
# 1. `MirroredStrategy`로 MNIST를 훈련하는 [튜토리얼](../tutorials/distribute/training_loops.ipynb).
# 2. `MirroredStrategy`를 사용하는 [DenseNet](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/densenet/distributed_train.py) 예제.
# 3. `MirroredStrategy`와 `TPUStrategy`를 사용하여 훈련하는 [BERT](https://github.com/tensorflow/models/blob/master/official/bert/run_classifier.py) 예제.
# 이 예제는 분산 훈련 도중 체크포인트로부터 불러오거나 주기적인 체크포인트를 만드는 방법을 이해하는 데 매우 유용합니다.
# 4. `keras_use_ctl` 플래그를 켜서 활성화할 수 있는 `MirroredStrategy`로 훈련한 [NCF](https://github.com/tensorflow/models/blob/master/official/recommendation/ncf_keras_main.py) 예제.
# 5. `MirroredStrategy`를 사용하여 훈련하는 [NMT](https://github.com/tensorflow/examples/blob/master/tensorflow_examples/models/nmt_with_attention/distributed_train.py) 예제.

# ## 추정기(Estimator)와 함께 `tf.distribute.Strategy` 사용하기
# `tf.estimator`는 원래부터 비동기 파라미터 서버 방식을 지원했던 분산 훈련 텐서플로 API입니다. 케라스와 마찬가지로 `tf.distribute.Strategy`를 `tf.estimator`와 함께 쓸 수 있습니다. 추정기 사용자는 아주 조금만 코드를 변경하면, 훈련이 분산되는 방식을 쉽게 바꿀 수 있습니다. 따라서 이제는 추정기 사용자들도 다중 GPU나 다중 워커뿐 아니라 다중 TPU에서 동기 방식으로 분산 훈련을 할 수 있습니다. 하지만 추정기는 제한적으로 지원하는 것입니다. 자세한 내용은 아래 [현재 어떤 것이 지원됩니까?](#estimator_support) 부분을 참고하십시오.
# 
# 추정기와 함께 `tf.distribute.Strategy`를 사용하는 방법은 케라스와는 살짝 다릅니다. `strategy.scope`를 사용하는 대신에, 전략 객체를 추정기의 [`RunConfig`](https://www.tensorflow.org/api_docs/python/tf/estimator/RunConfig)(실행 설정)에 넣어서 전달해야합니다.
# 
# 다음은 기본으로 제공되는 `LinearRegressor`와 `MirroredStrategy`를 함께 사용하는 방법을 보여주는 코드입니다.

# In[ ]:


mirrored_strategy = tf.distribute.MirroredStrategy()
config = tf.estimator.RunConfig(
    train_distribute=mirrored_strategy, eval_distribute=mirrored_strategy)
regressor = tf.estimator.LinearRegressor(
    feature_columns=[tf.feature_column.numeric_column('feats')],
    optimizer='SGD',
    config=config)


# 위 예제에서는 기본으로 제공되는 추정기를 사용하였지만, 직접 만든 추정기도 동일한 코드로 사용할 수 있습니다. `train_distribute`가 훈련을 어떻게 분산시킬지를 지정하고, `eval_distribute`가 평가를 어떻게 분산시킬지를 지정합니다. 케라스와 함께 사용할 때 훈련과 평가에 동일한 분산 전략을 사용했던 것과는 차이가 있습니다.
# 
# 다음과 같이 입력 함수를 지정하면 추정기의 훈련과 평가를 할 수 있습니다.

# In[ ]:


def input_fn():
  dataset = tf.data.Dataset.from_tensors(({"feats":[1.]}, [1.]))
  return dataset.repeat(1000).batch(10)
regressor.train(input_fn=input_fn, steps=10)
regressor.evaluate(input_fn=input_fn, steps=10)


# 추정기와 케라스의 또 다른 점인 입력 처리 방식을 살펴봅시다. 케라스에서는 각 배치의 데이터가 여러 개의 복제된 작업으로 나누어진다고 설명했습니다. 하지만 추정기에서는 사용자가 `input_fn` 입력 함수를 제공하고, 데이터를 워커나 장비들에 어떻게 나누어 처리할지를 온전히 제어할 수 있습니다. 텐서플로는 배치의 데이터를 자동으로 나누지도 않고, 각 워커에 자동으로 분배하지도 않습니다. 제공된 `input_fn` 함수는 워커마다 한 번씩 호출됩니다. 따라서 워커마다 데이터셋을 받게 됩니다. 한 데이터셋의 배치 하나가 워커의 복제된 작업 하나에 들어가고, 따라서 워커 하나에 N개의 복제된 작업이 있으면 N개의 배치가 수행됩니다. 다시 말하자면 `input_fn`이 반환하는 데이터셋은 `PER_REPLICA_BATCH_SIZE` 즉 복제 작업 하나가 배치 하나에서 처리할 크기여야 합니다. 한 단계에서 처리하는 전체 배치 크기는 `PER_REPLICA_BATCH_SIZE * strategy.num_replicas_in_sync`가 됩니다. 다중 워커를 사용하여 훈련할 때는 데이터를 워커별로 쪼개거나, 아니면 각자 다른 임의의 순서로 섞는 것이 좋을 수도 있습니다. 이렇게 처리하는 예제는 [추정기로 다중 워커를 써서 훈련하기](../tutorials/distribute/multi_worker_with_estimator.ipynb)에서 볼 수 있습니다.

# 추정기와 함께 `MirroredStrategy`를 사용하는 예를 보았습니다. `TPUStrategy`도 같은 방법으로 추정기와 함께 사용할 수 있습니다.
# ```
# config = tf.estimator.RunConfig(
#     train_distribute=tpu_strategy, eval_distribute=tpu_strategy)
# ```

# 비슷하게 다중 워커나 파라미터 서버를 사용한 전략도 사용할 수 있습니다. 코드는 거의 같지만, `tf.estimator.train_and_evaluate`를 사용해야 합니다. 그리고 클러스터에서 프로그램을 실행할 때 "TF_CONFIG" 환경변수를 설정해야 합니다.

# ### 현재 어떤 것이 지원됩니까?
# 
# TF 2.0 베타 버전에서는 추정기와 함께 모든 전략을 제한적으로 지원합니다. 기본적인 훈련과 평가는 동작합니다. 하지만 스캐폴드(scaffold) 같은 고급 기능은 아직 동작하지 않습니다. 또한 다소 버그가 있을 수 있습니다. 현재로써는 추정기와 함께 사용하는 것을 활발히 개선할 계획은 없습니다. 대신 케라스나 사용자 정의 훈련 루프 지원에 집중할 계획입니다. 만약 가능하다면 `tf.distribute` 사용시 이 API들을 먼저 고려하여 주십시오.
# 
# | 훈련 API  	| MirroredStrategy 	| TPUStrategy 	| MultiWorkerMirroredStrategy 	| CentralStorageStrategy 	| ParameterServerStrategy 	|
# |:---------------	|:------------------	|:-------------	|:-----------------------------	|:------------------------	|:-------------------------	|
# | 추정기 API 	| 제한적으로 지원 | 제한적으로 지원 | 제한적으로 지원 | 제한적으로 지원 | 제한적으로 지원 |
# 
# ### 예제와 튜토리얼
# 다음은 추정기와 함께 다양한 전략을 사용하는 방법을 처음부터 끝까지 보여주는 예제들입니다.
# 
# 1. [추정기로 다중 워커를 써서 훈련하기](../tutorials/distribute/multi_worker_with_estimator.ipynb)에서는 `MultiWorkerMirroredStrategy`로 다중 워커를 써서 MNIST를 훈련합니다.
# 2. [처음부터 끝까지 살펴보는 예제](https://github.com/tensorflow/ecosystem/tree/master/distribution_strategy)에서는 tensorflow/ecosystem의 쿠버네티스(Kubernetes) 템플릿을 이용하여 다중 워커를 사용하여 훈련합니다. 이 예제에서는 케라스 모델로 시작해서 `tf.keras.estimator.model_to_estimator` API를 이용하여 추정기 모델로 변환합니다.
# 3. `MirroredStrategy`나 `MultiWorkerMirroredStrategy`로 훈련할 수 있는 공식 [ResNet50](https://github.com/tensorflow/models/blob/master/official/r1/resnet/imagenet_main.py) 모델.
# 4. `TPUStrategy`를 사용한 [ResNet50](https://github.com/tensorflow/tpu/blob/master/models/experimental/distribution_strategy/resnet_estimator.py) 예제.

# ## 그 밖의 주제
# 이번 절에서는 다양한 사용 방식에 관련한 몇 가지 주제들을 다룹니다.

# <a id="TF_CONFIG">
# ### TF\_CONFIG 환경변수 설정하기
# </a>
# 
# 다중 워커를 사용하여 훈련할 때는, 앞서 설명했듯이 클러스터의 각 실행 프로그램마다 "TF\_CONFIG" 환경변수를 설정해야합니다. "TF\_CONFIG" 환경변수는 JSON 형식입니다. 그 안에는 클러스터를 구성하는 작업과 작업의 주소 및 각 작업의 역할을 기술합니다. [tensorflow/ecosystem](https://github.com/tensorflow/ecosystem) 저장소에서 훈련 작업에 맞게 "TF\_CONFIG"를 설정하는 쿠버네티스(Kubernetes) 템플릿을 제공합니다.
# 
# "TF\_CONFIG" 예를 하나 보면 다음과 같습니다.
# ```
# os.environ["TF_CONFIG"] = json.dumps({
#     "cluster": {
#         "worker": ["host1:port", "host2:port", "host3:port"],
#         "ps": ["host4:port", "host5:port"]
#     },
#    "task": {"type": "worker", "index": 1}
# })
# ```
# 

# 이 "TF\_CONFIG"는 세 개의 워커와 두 개의 파라미터 서버(ps) 작업을 각각의 호스트 및 포트와 함께 지정하고 있습니다. "task" 부분은 클러스터 내에서 현재 작업이 담당한 역할을 지정합니다. 여기서는 워커(worker) 1번, 즉 두 번째 워커라는 뜻입니다. 클러스터 내에서 가질 수 있는 역할은 "chief"(지휘자), "worker"(워커), "ps"(파라미터 서버), "evaluator"(평가자) 중 하나입니다. 단, "ps" 역할은 `tf.distribute.experimental.ParameterServerStrategy` 전략을 사용할 때만 쓸 수 있습니다.

# ## 다음으로는...
# 
# `tf.distribute.Strategy`는 활발하게 개발 중입니다. 한 번 써보시고 [깃허브 이슈](https://github.com/tensorflow/tensorflow/issues/new)를 통하여 피드백을 주시면 감사하겠습니다.
