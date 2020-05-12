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


# # 비정형 텐서
# 
# <table class="tfo-notebook-buttons" align="left">
#   <td>
#     <a target="_blank" href="https://www.tensorflow.org/guide/ragged_tensor"><img src="https://www.tensorflow.org/images/tf_logo_32px.png" />TensorFlow.org에서 보기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://colab.research.google.com/github/tensorflow/docs-l10n/blob/master/site/ko/guide/ragged_tensor.ipynb"><img src="https://www.tensorflow.org/images/colab_logo_32px.png" />구글 코랩(Colab)에서 실행하기</a>
#   </td>
#   <td>
#     <a target="_blank" href="https://github.com/tensorflow/docs-l10n/blob/master/site/ko/guide/ragged_tensor.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png" />깃허브(GitHub)에서 소스 보기</a>
#   </td>
#   <td>
#     <a href="https://storage.googleapis.com/tensorflow_docs/docs-l10n/site/ko/guide/ragged_tensor.ipynb"><img src="https://www.tensorflow.org/images/download_logo_32px.png" />노트북 다운로드</a>
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

# In[ ]:


import math
import tensorflow as tf


# ## 개요
# 
# 데이터는 다양한 형태로 제공됩니다; 텐서도 마찬가지입니다.
# *비정형 텐서*는 중첩 가변 길이 목록에 해당하는 텐서플로입니다.
# 다음을 포함하여 균일하지 않은 모양으로 데이터를 쉽게 저장하고 처리할 수 있습니다:
# 
# 
# *   일련의 영화의 배우들과 같은 가변 길이 기능
# *   문장이나 비디오 클립과 같은 가변 길이 순차적 입력의 배치
#     
# *   절, 단락, 문장 및 단어로 세분화된 텍스트 문서와 같은 계층적 입력
#     
# *   프로토콜 버퍼와 같은 구조화된 입력의 개별 필드
# 
# ### 비정형 텐서로 할 수 있는 일
# 
# 비정형 텐서는 수학 연산 (예 : `tf.add` 및 `tf.reduce_mean`), 
# 배열 연산 (예 : `tf.concat` 및 `tf.tile`),
# 문자열 조작 작업 (예 : `tf.substr`)을 포함하여 수백 가지 이상의 텐서플로 연산에서 지원됩니다
# :
# 

# In[ ]:


digits = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
words = tf.ragged.constant([["So", "long"], ["thanks", "for", "all", "the", "fish"]])
print(tf.add(digits, 3))
print(tf.reduce_mean(digits, axis=1))
print(tf.concat([digits, [[5, 3]]], axis=0))
print(tf.tile(digits, [1, 2]))
print(tf.strings.substr(words, 0, 2))


# 팩토리 메서드, 변환 메서드 및 값 매핑 연산을 포함하여 비정형 텐서에 
# 고유한 여러 메서드 및 연산도 있습니다.
# 
# 지원되는 작업 목록은 `tf.ragged` 패키지 문서를 참조하십시오.
# 
# 
# 일반 텐서와 마찬가지로, Python 스타일 인덱싱을 사용하여 비정형 텐서의 특정 부분에 접근할 수 있습니다.
# 자세한 내용은 아래
# **인덱싱** 절을 참조하십시오.

# In[ ]:


print(digits[0])       # 첫 번째 행


# In[ ]:


print(digits[:, :2])   # 각 행의 처음 두 값


# In[ ]:


print(digits[:, -2:])  # 각 행의 마지막 두 값


# 일반 텐서와 마찬가지로, 파이썬 산술 및 비교 연산자를 사용하여 요소 별 연산을 수행할 수 있습니다.
# 자세한 내용은 아래의
# **오버로드된 연산자** 절을 참조하십시오.

# In[ ]:


print(digits + 3)


# In[ ]:


print(digits + tf.ragged.constant([[1, 2, 3, 4], [], [5, 6, 7], [8], []]))


# `RaggedTensor`의 값으로 요소 별 변환을 수행해야하는 경우, 함수와 하나 이상의 매개변수를 갖는 `tf.ragged.map_flat_values`를 사용할 수 있고, `RaggedTensor`의 값을 변환할 때 적용할 수 있습니다.

# In[ ]:


times_two_plus_one = lambda x: x * 2 + 1
print(tf.ragged.map_flat_values(times_two_plus_one, digits))


# ### 비정형 텐서 생성하기
# 
# 비정형 텐서를 생성하는 가장 간단한 방법은 
# `tf.ragged.constant`를 사용하는 것입니다. `tf.ragged.constant`는 주어진 중첩된 Python 목록에 해당하는 `RaggedTensor`를
# 빌드 합니다:

# In[ ]:


sentences = tf.ragged.constant([
    ["Let's", "build", "some", "ragged", "tensors", "!"],
    ["We", "can", "use", "tf.ragged.constant", "."]])
print(sentences)


# In[ ]:


paragraphs = tf.ragged.constant([
    [['I', 'have', 'a', 'cat'], ['His', 'name', 'is', 'Mat']],
    [['Do', 'you', 'want', 'to', 'come', 'visit'], ["I'm", 'free', 'tomorrow']],
])
print(paragraphs)


# 비정형 텐서는 `tf.RaggedTensor.from_value_rowids`, `tf.RaggedTensor.from_row_lengths` 및 `tf.RaggedTensor.from_row_splits`와 
# `tf.RaggedTensor.from_row_splits`와 같은 팩토리 클래스 메서드를 사용하여
# 플랫 *values* 텐서와 *행 분할* 텐서를 쌍을 지어 해당 값을 행으로 분할하는 방법을 표시하는 방식으로도 생성할 수 있습니다.
# 
# 
# 
# #### `tf.RaggedTensor.from_value_rowids`
# 각 값이 속하는 행을 알고 있으면 `value_rowids` 행 분할 텐서를 사용하여 `RaggedTensor`를 빌드할 수 있습니다:
# 
# ![value_rowids](https://www.tensorflow.org/images/ragged_tensors/value_rowids.png)

# In[ ]:


print(tf.RaggedTensor.from_value_rowids(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    value_rowids=[0, 0, 0, 0, 2, 2, 2, 3]))


# #### `tf.RaggedTensor.from_row_lengths`
# 
# 각 행의 길이를 알고 있으면 `row_lengths` 행 분할 텐서를 사용할 수 있습니다:
# 
# ![row_lengths](https://www.tensorflow.org/images/ragged_tensors/row_lengths.png)

# In[ ]:


print(tf.RaggedTensor.from_row_lengths(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    row_lengths=[4, 0, 3, 1]))


# #### `tf.RaggedTensor.from_row_splits`
# 
# 각 행의 시작과 끝 인덱스를 알고 있다면 `row_splits` 행 분할 텐서를 사용할 수 있습니다:
# 
# ![row_splits](https://www.tensorflow.org/images/ragged_tensors/row_splits.png)

# In[ ]:


print(tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2, 6],
    row_splits=[0, 4, 4, 7, 8]))


# 팩토리 메서드의 전체 목록은 `tf.RaggedTensor` 클래스 문서를 참조하십시오.

# ### 비정형 텐서에 저장할 수 있는 것
# 
# 일반 `텐서`와 마찬가지로, `RaggedTensor`의 값은 모두 같은 유형이어야 합니다;
# 값은 모두 동일한 중첩 깊이 (텐서의 *랭크*)에
# 있어야 합니다:

# In[ ]:


print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]))  # 좋음: 유형=문자열, 랭크=2


# In[ ]:


print(tf.ragged.constant([[[1, 2], [3]], [[4, 5]]]))        # 좋음: 유형=32비트정수, 랭크=3


# In[ ]:


try:
  tf.ragged.constant([["one", "two"], [3, 4]])              # 안좋음: 다수의 유형
except ValueError as exception:
  print(exception)


# In[ ]:


try:
  tf.ragged.constant(["A", ["B", "C"]])                     # 안좋음: 다중첩 깊이
except ValueError as exception:
  print(exception)


# ### 사용 예시
# 
# 다음 예제는 `RaggedTensor`를 사용하여 각 문장의 시작과 끝에 특수 마커를 사용하여
# 가변 길이 쿼리 배치에 대한 유니그램 및 바이그램 임베딩을 생성하고 결합하는 방법을 보여줍니다.
# 이 예제에서 사용된 작업에 대한 자세한 내용은
# `tf.ragged` 패키지 설명서를 참조하십시오.

# In[ ]:


queries = tf.ragged.constant([['Who', 'is', 'Dan', 'Smith'],
                              ['Pause'],
                              ['Will', 'it', 'rain', 'later', 'today']])

# 임베딩 테이블 만들기
num_buckets = 1024
embedding_size = 4
embedding_table = tf.Variable(
    tf.random.truncated_normal([num_buckets, embedding_size],
                       stddev=1.0 / math.sqrt(embedding_size)))

# 각 단어에 대한 임베딩 찾기
word_buckets = tf.strings.to_hash_bucket_fast(queries, num_buckets)
word_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, word_buckets)                  # ①

# 각 문장의 시작과 끝에 마커 추가하기
marker = tf.fill([queries.nrows(), 1], '#')
padded = tf.concat([marker, queries, marker], axis=1)                       # ②

# 바이그램 빌드 & 임베딩 찾기
bigrams = tf.strings.join([padded[:, :-1],
                               padded[:, 1:]],
                              separator='+')                                # ③

bigram_buckets = tf.strings.to_hash_bucket_fast(bigrams, num_buckets)
bigram_embeddings = tf.ragged.map_flat_values(
    tf.nn.embedding_lookup, embedding_table, bigram_buckets)                # ④

# 각 문장의 평균 임베딩 찾기
all_embeddings = tf.concat([word_embeddings, bigram_embeddings], axis=1)    # ⑤
avg_embedding = tf.reduce_mean(all_embeddings, axis=1)                      # ⑥
print(avg_embedding)


# ![ragged_example](https://www.tensorflow.org/images/ragged_tensors/ragged_example.png)

# ## 비정형 텐서: 정의
# 
# ### 비정형 및 정형 차원
# 
# *비정형 텐서*는 슬라이스의 길이가 다를 수 있는 하나 이상의 *비정형 크기*를 갖는 텐서입니다.
# 예를 들어, 
# `rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []]` 의 내부 (열) 크기는 
# 열 슬라이스(`rt[0, :]`, ..., `rt[4, :]`)의 길이가 다르기 때문에 비정형입니다.
# 부분의 길이가 모두 같은 차원을 *정형차원*이라고 합니다.
# 
# 비정형 텐서의 가장 바깥 쪽 차원은 단일 슬라이스로 구성되므로 슬라이스의 길이가
# 다를 가능성이 없으므로 항상 균일합니다.
# 비정형 텐서는 균일한 가장 바깥 쪽 차원에 더하여 균일한 내부 차원을 가질 수도 있습니다.
# 예를 들어, `[num_sentences, (num_words), embedding_size]` 형태의 비정형 텐서를 사용하여 
# 각 단어에 대한 단어 임베딩을 일련의 문장으로 저장할 수 있습니다.
# 여기서 `(num_words)`의 괄호는 차원이 비정형임을 나타냅니다.
# 
# ![sent_word_embed](https://www.tensorflow.org/images/ragged_tensors/sent_word_embed.png)
# 
# 비정형 텐서는 다수의 비정형 차원을 가질 수 있습니다. 예를 들어
# 모양이 `[num_documents, (num_paragraphs), (num_sentences), (num_words)]` 인
#  텐서를 사용하여 일련의 구조화된 텍스트 문서를 저장할 수 있습니다.
#  (여기서 괄호는 비정형 차원임을 나타냅니다.)
# 
# #### 비정형 텐서 형태 제한
# 
# 비정형 텐서의 형태는 다음과 같은 형식으로 제한됩니다:
# 
# *   단일 정형 차원
# *   하나 이상의 비정형 차원
# *   0 또는 그 이상의 정형 차원
# 
# 참고: 이러한 제한은 현재 구현의 결과이며
# 향후 완화될 수 있습니다.
# 
# ### 랭크 및 비정형 랭크
# 
# 비정형 텐서의 총 차원 수를 ***랭크***라고 하고,
# 비정형 텐서의 비정형 차원 수를 ***비정형랭크***라고 합니다. 그래프 실행 모드 (즉, 비 즉시 실행(non-eager) 모드)에서, 텐서의 비정형 랭크는
# 생성 시 고정됩니다: 비정형 랭크는 런타임 값에 의존할 수 없으며 다른 세션 실행에 따라
# 동적으로 변할 수 없습니다.
# ***잠재적으로 비정형인 텐서***는
# `tf.Tensor` 또는 `tf.RaggedTensor` 일 수 있는 값입니다.
# `tf.Tensor`의 비정형 랭크는 0으로 정의됩니다.
# 
# ### 비정형 텐서 형태
# 
# 비정형 텐서의 형태를 설명할 때, 비정형 차원은 괄호로 묶어 표시됩니다.
# 예를 들어, 위에서 살펴본 것처럼 일련의 문장에서 각 단어에 대한 단어 임베딩을 저장하는
# 3차원 비정형텐서의 형태는
# `[num_sentences, (num_words), embedding_size]`로 나타낼 수 있습니다.
# `RaggedTensor.shape` 프로퍼티는 비정형 텐서에 대해 크기가 없는 비정형 차원인 `tf.TensorShape`를 반환합니다:
# 

# In[ ]:


tf.ragged.constant([["Hi"], ["How", "are", "you"]]).shape


# `tf.RaggedTensor.bounding_shape` 메서드를 사용하여 지정된
# `RaggedTensor`에 대한 빈틈이 없는 경계 형태를 찾을 수 있습니다:

# In[ ]:


print(tf.ragged.constant([["Hi"], ["How", "are", "you"]]).bounding_shape())


# ## 비정형 vs 희소 텐서
# 
# 비정형텐서는 희소 텐서의 유형이 *아니라*
# 불규칙한 형태의 밀집 텐서로 간주되어야 합니다.
# 
# 예를 들어, 비정형 vs 희소 텐서에 대해 `concat`,
# `stack` 및 `tile`과 같은 배열 연산이 어떻게 정의되는지 고려하십시오.
# 비정형 텐서들을 연결하면 각 행을 결합하여 단일 행을 형성합니다:
# 
# ![ragged_concat](https://www.tensorflow.org/images/ragged_tensors/ragged_concat.png)
# 

# In[ ]:


ragged_x = tf.ragged.constant([["John"], ["a", "big", "dog"], ["my", "cat"]])
ragged_y = tf.ragged.constant([["fell", "asleep"], ["barked"], ["is", "fuzzy"]])
print(tf.concat([ragged_x, ragged_y], axis=1))


# 그러나 희소 텐서를 연결하는 것은 다음 예에 표시된 것처럼
#  해당 밀집 텐서를 연결하는 것과 같습니다. (여기서 Ø는 누락된 값을 나타냅니다.):
# 
# ![희소 텐서 합치기](https://www.tensorflow.org/images/ragged_tensors/sparse_concat.png)
# 

# In[ ]:


sparse_x = ragged_x.to_sparse()
sparse_y = ragged_y.to_sparse()
sparse_result = tf.sparse.concat(sp_inputs=[sparse_x, sparse_y], axis=1)
print(tf.sparse.to_dense(sparse_result, ''))


# 이 구별이 중요한 이유의 다른 예를 보려면,
# `tf.reduce_mean`과 같은 연산에 대한 “각 행의 평균값”의 정의를 고려하십시오.
# 비정형 텐서의 경우, 행의 평균값은 행 값을 행 너비로 나눈 값의 합입니다.
# 그러나 희소 텐서의 경우 행의 평균값은
# 행 값의 합계롤 희소 텐서의 전체 너비(가장 긴 행의 너비 이상)로
# 나눈 값입니다.
# 

# ## 오버로드된 연산자
# 
# `RaggedTensor` 클래스는 표준 Python 산술 및 비교 연산자를 오버로드하여
# 기본 요소 별 수학을 쉽게 수행할 수 있습니다:

# In[ ]:


x = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
y = tf.ragged.constant([[1, 1], [2], [3, 3, 3]])
print(x + y)


# 오버로드된 연산자는 요소 단위 계산을 수행하므로, 모든
# 이진 연산에 대한 입력은 동일한 형태이거나, 동일한 형태로 브로드캐스팅 할 수 있어야 합니다.
# 가장 간단한 확장의 경우, 단일 스칼라가 비정형 텐서의
# 각 값과 요소 별로 결합됩니다:

# In[ ]:


x = tf.ragged.constant([[1, 2], [3], [4, 5, 6]])
print(x + 3)


# 고급 사례에 대한 설명은 **브로드캐스팅** 절을
# 참조하십시오.
# 
# 비정형 텐서는 일반 `텐서`와 동일한 연산자 세트를 오버로드합니다:단항
# 연산자 `-`, `~` 및 `abs()`; 그리고 이항 연산자 `+`, `-`, `*`, `/`,
# `//`, `%`, `**`, `&`, `|`, `^`, `==`, `<`, `<=`, `>` 및 `>=`.
# 

# ## 인덱싱
# 
# 비정형 텐서는 다차원 인덱싱 및 슬라이싱을 포함하여 Python 스타일 인덱싱을 지원합니다.
# 다음 예는 2차원 및 3차원 비정형 텐서를 사용한 비정형 텐서 인덱싱을
# 보여줍니다.
# 
# ### 비정형 1차원으로 2차원 비정형 텐서 인덱싱

# In[ ]:


queries = tf.ragged.constant(
    [['Who', 'is', 'George', 'Washington'],
     ['What', 'is', 'the', 'weather', 'tomorrow'],
     ['Goodnight']])
print(queries[1])


# In[ ]:


print(queries[1, 2])                # 한 단어


# In[ ]:


print(queries[1:])                  # 첫 번째 행을 제외한 모든 단어


# In[ ]:


print(queries[:, :3])               # 각 쿼리의 처음 세 단어


# In[ ]:


print(queries[:, -2:])              # 각 쿼리의 마지막 두 단어


# ### 비정형 2차원으로 3차원 비정형 텐서 인덱싱

# In[ ]:


rt = tf.ragged.constant([[[1, 2, 3], [4]],
                         [[5], [], [6]],
                         [[7]],
                         [[8, 9], [10]]])


# In[ ]:


print(rt[1])                        # 두 번째 행 (2차원 비정형 텐서)


# In[ ]:


print(rt[3, 0])                     # 네 번째 행의 첫 번째 요소 (1차원 텐서)


# In[ ]:


print(rt[:, 1:3])                   # 각 행의 1-3 항목 (3차원 비정형 텐서)


# In[ ]:


print(rt[:, -1:])                   # 각 행의 마지막 항목 (3차원 비정형 텐서)


# `RaggedTensor`는 다차원 인덱싱 및 슬라이싱을 지원하며, 한 가지 제한 사항이
# 있습니다: 비정형 차원으로 인덱싱할 수 없습니다. 이 값은
# 표시된 값이 일부 행에 존재할 수 있지만 다른 행에는 존재하지 않기 때문에 문제가 됩니다.
# 그러한 경우, 우리가 (1) `IndexError`를 제기해야 하는지; (2)
# 기본값을 사용해야 하는지; 또는 (3) 그 값을 스킵하고 시작한 것보다 적은 행을 가진 텐서를 반환해야 하는지
# 에 대한 여부는 확실하지 않습니다.
# [Python의 안내 지침](https://www.python.org/dev/peps/pep-0020/)
# ("애매한 상황에서
# 추측하려고 하지 마십시오" )에 따라, 현재 이 작업을 허용하지
# 않습니다.

# ## 텐서 형 변환
# 
# `RaggedTensor` 클래스는
# `RaggedTensor`와 `tf.Tensor` 또는 `tf.SparseTensors` 사이를 변환하는데 사용할 수 있는 메서드를 정의합니다:

# In[ ]:


ragged_sentences = tf.ragged.constant([
    ['Hi'], ['Welcome', 'to', 'the', 'fair'], ['Have', 'fun']])
print(ragged_sentences.to_tensor(default_value=''))


# In[ ]:


print(ragged_sentences.to_sparse())


# In[ ]:


x = [[1, 3, -1, -1], [2, -1, -1, -1], [4, 5, 8, 9]]
print(tf.RaggedTensor.from_tensor(x, padding=-1))


# In[ ]:


st = tf.SparseTensor(indices=[[0, 0], [2, 0], [2, 1]],
                     values=['a', 'b', 'c'],
                     dense_shape=[3, 3])
print(tf.RaggedTensor.from_sparse(st))


# ## 비정형 텐서 평가
# 
# ### 즉시 실행
# 
# 즉시 실행 모드에서는, 비정형 텐서가 즉시 실행됩니다. 포함된 값에
# 접근하려면 다음을 수행하십시오:
# 
# *   비정형 텐서를 Python `목록`으로 변환하는
#     `tf.RaggedTensor.to_list()`
#     메서드를 사용하십시오.

# In[ ]:


rt = tf.ragged.constant([[1, 2], [3, 4, 5], [6], [], [7]])
print(rt.to_list())


# *   Python 인덱싱을 사용하십시오. 선택한 텐서 조각에 비정형 차원이 없으면,
#     `EagerTensor`로 반환됩니다. 그런 다음
#     `numpy()`메서드를 사용하여 값에 직접 접근할 수 있습니다.

# In[ ]:


print(rt[1].numpy())


# *   `tf.RaggedTensor.values` 및
#     `tf.RaggedTensor.row_splits` 특성 또는
#     `tf.RaggedTensor.row_lengths()` 및
#     `tf.RaggedTensor.value_rowids()`와 같은 행 분할 메서드를 사용하여
#     비정형 텐서를 구성 요소로
#     분해하십시오.

# In[ ]:


print(rt.values)


# In[ ]:


print(rt.row_splits)


# ### 브로드캐스팅
# 
# 브로드캐스팅은 다른 형태의 텐서가 요소 별 연산에 적합한 형태를 갖도록 만드는 프로세스입니다.
# 브로드캐스팅에 대한 자세한 내용은
# 다음을 참조하십시오:
# 
# *   [Numpy: 브로드캐스팅](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
# *   `tf.broadcast_dynamic_shape`
# *   `tf.broadcast_to`
# 
# 호환 가능한 형태를 갖도록 두 개의 입력 `x` 와 `y` 를 브로드캐스팅하는 기본 단계는
# 다음과 같습니다:
# 
# 1.  `x` 와 `y` 의 차원 수가 동일하지 않은 경우, 외부 차원
#     (크기 1)을 차원 수가 동일해질 때까지 추가합니다 .
# 
# 2.  `x` 와 `y` 의 크기가 다른 각 차원에 대해:
# 
#     *   차원 `d`에 `x` 또는 `y`의 크기가 `1` 이면, 다른 입력의 크기와 일치하도록
#         차원 `d`에서 값을 반복하십시오.
# 
#     *   그렇지 않으면 예외가 발생합니다 (`x` 와 `y` 는 브로드캐스트와 호환되지
#         않습니다).

# 정형 차원에서 텐서의 크기가 단일 숫자 (해당 차원에서
# 슬라이스 크기)인 경우; 그리고 비정형 차원에서 텐서의 크기가 슬라이스 길이의 목록인 경우
# (해당 차원의 모든 슬라이스에 대해).
# 
# #### 브로드캐스팅 예제

# In[ ]:


# x       (2D ragged):  2 x (num_rows)
# y       (scalar)
# 결과     (2D ragged):  2 x (num_rows)
x = tf.ragged.constant([[1, 2], [3]])
y = 3
print(x + y)


# In[ ]:


# x         (2d ragged):  3 x (num_rows)
# y         (2d tensor):  3 x          1
# 결과       (2d ragged):  3 x (num_rows)
x = tf.ragged.constant(
   [[10, 87, 12],
    [19, 53],
    [12, 32]])
y = [[1000], [2000], [3000]]
print(x + y)


# In[ ]:


# x      (3d ragged):  2 x (r1) x 2
# y      (2d ragged):         1 x 1
# 결과    (3d ragged):  2 x (r1) x 2
x = tf.ragged.constant(
    [[[1, 2], [3, 4], [5, 6]],
     [[7, 8]]],
    ragged_rank=1)
y = tf.constant([[10]])
print(x + y)


# In[ ]:


# x      (3d ragged):  2 x (r1) x (r2) x 1
# y      (1d tensor):                    3
# 결과    (3d ragged):  2 x (r1) x (r2) x 3
x = tf.ragged.constant(
    [
        [
            [[1], [2]],
            [],
            [[3]],
            [[4]],
        ],
        [
            [[5], [6]],
            [[7]]
        ]
    ],
    ragged_rank=2)
y = tf.constant([10, 20, 30])
print(x + y)


# 브로드캐스트 하지 않는 형태의 예는 다음과 같습니다:

# In[ ]:


# x      (2d ragged): 3 x (r1)
# y      (2d tensor): 3 x    4  # 뒤의 차원은 일치하지 않습니다.
x = tf.ragged.constant([[1, 2], [3, 4, 5, 6], [7]])
y = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)


# In[ ]:


# x      (2d ragged): 3 x (r1)
# y      (2d ragged): 3 x (r2)  # 비정형 차원은 일치하지 않습니다.
x = tf.ragged.constant([[1, 2, 3], [4], [5, 6]])
y = tf.ragged.constant([[10, 20], [30, 40], [50]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)


# In[ ]:


# x      (3d ragged): 3 x (r1) x 2
# y      (3d ragged): 3 x (r1) x 3  # 뒤의 차원은 일치하지 않습니다.
x = tf.ragged.constant([[[1, 2], [3, 4], [5, 6]],
                        [[7, 8], [9, 10]]])
y = tf.ragged.constant([[[1, 2, 0], [3, 4, 0], [5, 6, 0]],
                        [[7, 8, 0], [9, 10, 0]]])
try:
  x + y
except tf.errors.InvalidArgumentError as exception:
  print(exception)


# ## RaggedTensor 인코딩
# 
# 비정형텐서는 `RaggedTensor` 클래스를 사용하여 인코딩됩니다. 내부적으로, 각
# `RaggedTensor`는 다음으로 구성됩니다:
# 
# *   가변 길이 행을 병합된 목록으로 연결하는 `values`
#     텐서
# *   병합된 값을 행으로 나누는 방법을 나타내는 `row_splits` 벡터,
#     특히, 행 `rt[i]`의 값은 슬라이스
#     `rt.values[rt.row_splits[i]:rt.row_splits[i+1]]`에 저장됩니다.
# 
# ![ragged_encoding](https://www.tensorflow.org/images/ragged_tensors/ragged_encoding.png)
# 

# In[ ]:


rt = tf.RaggedTensor.from_row_splits(
    values=[3, 1, 4, 1, 5, 9, 2],
    row_splits=[0, 4, 4, 6, 7])
print(rt)


# ### 다수의 비정형 차원
# 
# 다수의 비정형 차원을 갖는 비정형 텐서는
# `values` 텐서에 대해 중첩된 `RaggedTensor`를 사용하여 인코딩됩니다. 중첩된 각 `RaggedTensor`는
# 단일 비정형 차원을 추가합니다.
# 
# ![ragged_rank_2](https://www.tensorflow.org/images/ragged_tensors/ragged_rank_2.png)

# In[ ]:


rt = tf.RaggedTensor.from_row_splits(
    values=tf.RaggedTensor.from_row_splits(
        values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        row_splits=[0, 3, 3, 5, 9, 10]),
    row_splits=[0, 1, 1, 5])
print(rt)
print("형태: {}".format(rt.shape))
print("비정형 텐서의 차원 : {}".format(rt.ragged_rank))


# 팩토리 함수 `tf.RaggedTensor.from_nested_row_splits`는
# `row_splits` 텐서 목록을 제공하여 다수의 비정형 차원으로 RaggedTensor를
# 직접 생성하는데 사용할 수 있습니다:

# In[ ]:


rt = tf.RaggedTensor.from_nested_row_splits(
    flat_values=[10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
    nested_row_splits=([0, 1, 1, 5], [0, 3, 3, 5, 9, 10]))
print(rt)


# ### 정형한 내부 차원
# 
# 내부 차원이 정형한 비정형 텐서는
# `values`에 다차원 `tf.Tensor`를 사용하여 인코딩됩니다.
# 
# ![uniform_inner](https://www.tensorflow.org/images/ragged_tensors/uniform_inner.png)

# In[ ]:


rt = tf.RaggedTensor.from_row_splits(
    values=[[1, 3], [0, 0], [1, 3], [5, 3], [3, 3], [1, 2]],
    row_splits=[0, 3, 4, 6])
print(rt)
print("형태: {}".format(rt.shape))
print("비정형 텐서의 차원 : {}".format(rt.ragged_rank))


# ### 대체 가능한 행 분할 방식
# 
# `RaggedTensor` 클래스는 `row_splits`를 기본 메커니즘으로 사용하여
# 값이 행으로 분할되는 방법에 대한 정보를 저장합니다. 그러나,
# `RaggedTensor`는 네 가지 대체 가능한 행 분할 방식을 지원하므로 데이터 형식에 따라 더 편리하게
# 사용할 수 있습니다.
# 내부적으로, `RaggedTensor`는 이러한 추가적인 방식을 사용하여 일부 컨텍스트에서 효율성을
# 향상시킵니다.
# 
# <dl>
#   <dt>행 길이</dt>
#     <dd>`row_lengths`는 `[nrows]`형태의 벡터로, 각 행의 길이를
#     지정합니다.</dd>
# 
#   <dt>행 시작</dt>
#     <dd>`row_starts`는 `[nrows]`형태의 벡터로, 각 행의 시작 오프셋을
#     지정합니다. `row_splits[:-1]`와 같습니다.</dd>
# 
#   <dt>행 제한</dt>
#     <dd>`row_limits`는 `[nrows]`형태의 벡터로, 각 행의 정지 오프셋을
#     지정합니다. `row_splits[1:]`와 같습니다.</dd>
# 
#   <dt>행 인덱스 및 행 수</dt>
#     <dd>`value_rowids`는 `[nvals]`모양의 벡터로, 값과 일대일로 대응되며
#     각 값의 행 인덱스를 지정합니다.
#     특히, `rt[row]`행은 `value_rowids[j]==row`인 `rt.values[j]`값으로 구성됩니다.
#     \`nrows`는
#     `RaggedTensor`의 행 수를 지정하는 정수입니다.
#     특히, `nrows`는 뒤의 빈 행을 나타내는데
#     사용됩니다.</dd>
# </dl>
# 
# 예를 들어, 다음과 같이 비정형 텐서는 동일합니다:

# In[ ]:


values = [3, 1, 4, 1, 5, 9, 2, 6]
print(tf.RaggedTensor.from_row_splits(values, row_splits=[0, 4, 4, 7, 8, 8]))
print(tf.RaggedTensor.from_row_lengths(values, row_lengths=[4, 0, 3, 1, 0]))
print(tf.RaggedTensor.from_row_starts(values, row_starts=[0, 4, 4, 7, 8]))
print(tf.RaggedTensor.from_row_limits(values, row_limits=[4, 4, 7, 8, 8]))
print(tf.RaggedTensor.from_value_rowids(
    values, value_rowids=[0, 0, 0, 0, 2, 2, 2, 3], nrows=5))


# RaggedTensor 클래스는 이러한 각 행 분할 텐서를 생성하는데 사용할 수 있는
# 메서드를 정의합니다.

# In[ ]:


rt = tf.ragged.constant([[3, 1, 4, 1], [], [5, 9, 2], [6], []])
print("      values: {}".format(rt.values))
print("  row_splits: {}".format(rt.row_splits))
print(" row_lengths: {}".format(rt.row_lengths()))
print("  row_starts: {}".format(rt.row_starts()))
print("  row_limits: {}".format(rt.row_limits()))
print("value_rowids: {}".format(rt.value_rowids()))


# (`tf.RaggedTensor.values`와 `tf.RaggedTensors.row_splits`는 프로퍼티이며, 나머지 행 분할 접근자는 모두 메서드입니다. 이는 `row_splits`가 기본 표현이고 다른 행 분할 텐서는 계산되어야함을 나타냅니다.)

# 서로 다른 행 분할 방식의 장점과 단점은
# 다음과 같습니다:
# 
# + **효율적인 인덱싱**:
#     `row_splits`, `row_starts` 및 `row_limits` 방식은 모두 비정형 텐서에
#     일정한 시간 인덱싱을 가능하게 합니다. `value_rowids`와
#      `row_lengths` 방식은 가능하지 않습니다.
# 
# + **작은 인코딩 크기**:
#     텐서의 크기는 값의 총 수에만 의존하기 때문에 빈 행이 많은 비정형 텐서를 저장할 때 `value_rowids`
#     방식이 더 효율적입니다.
#     반면, 다른 4개의 인코딩은 각 행에 대해 하나의 스칼라 값만 필요하므로
#     행이 긴 비정형 텐서를
#     저장할 때 더 효율적입니다.
# 
# + **효율적인 연결**:
#    두 개의 텐서가 함께 연결될 때 행 길이가 변경되지 않으므로
#    (행 분할 및 행 인덱스는 변경되므로)
#    비정형 텐서를 연결할 때 `row_lengths` 방식이 더 효율적입니다.
# 
# + **호환성**:
#     `value_rowids` 방식은 `tf.segment_sum`과 같은 연산에서 사용되는
#     [분할](../api_guides/python/math_ops.md#Segmentation)
#     형식과 일치합니다. `row_limits` 방식은
#     `tf.sequence_mask`와 같이 작업에서 사용하는 형식과 일치합니다.
