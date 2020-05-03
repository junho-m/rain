import os
from os.path import join
from tqdm.notebook import tqdm
import multiprocessing

import numpy as np
import pandas as pd

# Path
ROOT_DIR_PATH = '.'
TRAIN_DIR_PATH = join(ROOT_DIR_PATH, 'train')
TEST_DIR_PATH = join(ROOT_DIR_PATH, 'test')
TRAIN_FEATHER_PATH = join(ROOT_DIR_PATH, 'train.ftr')
TEST_FEATHER_PATH = join(ROOT_DIR_PATH, 'test.ftr')

# Column(feature) names
INPUT_COL = [f"temp{i}" for i in range(1, 10)] + ['type', 'long_GMI', 'lat_GMI', 'long_DPR', 'lat_DPR']
TARGET_COL = ['precipitation']
TEST_COL = INPUT_COL
TRAIN_COL = INPUT_COL + TARGET_COL

# Name column (optional)
IS_PAD_NAME_COL = False
NAME_COL = ['orbit', 'subset', 'pixel']
PIXEL_COL = np.arange(1, 1601)[:, None]

'''
def pad_name_cols(nd, file_name):
    orbit, subset = file_name.split('_')[1:]
    subset = subset[:2]
    nd = np.pad(nd, ((0, 0), (0, 1)), constant_values=int(orbit))
    nd = np.pad(nd, ((0, 0), (0, 1)), constant_values=int(subset))
    return np.c_[nd, PIXEL_COL]

def generate_ndarray_from_file_name(file_name, dir_path):
    file_path = join(dir_path, file_name)
    nd = np.load(file_path).astype(np.float32)  # 40 x 40 x ?
    dim = nd.shape[-1]
    nd = nd.reshape(-1, dim)                    # 1600    x ?
    if IS_PAD_NAME_COL:
        nd = pad_name_col(nd, file_name)
    return nd
    
def generate_ndarray_from_dir_path(dir_path):
    pool = multiprocessing.Pool()
    nds = pool.starmap(generate_ndarray_from_file_name, [(file_name, dir_path) for file_name in tqdm(os.listdir(dir_path))])
    return np.concatenate(nds)


def generate_dataframe_from_dir_path(dir_path):
    nd = generate_ndarray_from_dir_path(dir_path)
    dim = nd.shape[-1]
    df =  pd.DataFrame(nd,
                       columns=TRAIN_COL if dim == len(TRAIN_COL) else TEST_COL,
                       dtype=np.float32
                      )
    if IS_PAD_NAME_COL:
        df[['orbit', 'subset', 'pixel']] = df[['orbit', 'subset', 'pixel']].astype(np.int32)
        df.sort_values(by=['orbit', 'subset', 'pixel'], ignore_index=True, inplace=True)
    return df


def main():
    global TRAIN_COL, TEST_COL
    TRAIN_COL, TEST_COL = (TRAIN_COL, TEST_COL) if not IS_PAD_NAME_COL else (TRAIN_COL + NAME_COL, TEST_COL + NAME_COL)
    for dir_path in (TRAIN_DIR_PATH, TEST_DIR_PATH):
        df = generate_dataframe_from_dir_path(dir_path)
        if len(df.columns) == len(TRAIN_COL):
            train_df = df
        else:
            test_df = df
    return train_df, test_df

def to_feather(train_df, test_df):
    train_df.to_feather(TRAIN_FEATHER_PATH)
    test_df.to_feather(TEST_FEATHER_PATH)

def read_feather():
    train_df = pd.read_feather(TRAIN_FEATHER_PATH)
    test_df = pd.read_feather(TEST_FEATHER_PATH)
    return train_df, test_df

from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('load_ext', 'memory_profiler')

'''
#%load_ext memory_profiler
#%load_ext line_profiler

#%%time
#%memit train_df, test_df = main()
# 14s / 15GB

#ArithmeticErroroad_ext memory_profiler
#%load_ext line_profiler
#%%time
#%memit train_df, test_df = main()
#A 14s / 15GB
#HBox(children=(FloatProgress(value=0.0, max=76345.0), HTML(value='')))
#HBox(children=(FloatProgress(value=0.0, max=2416.0), HTML(value='')))
#peak memory: 14355.44 MiB, increment: 14259.02 MiB
#CPU times: user 3.75 s, sys: 7.17 s, total: 10.9 s
##Wall time: 14 s
#%%time
#%memit to_feather(train_df, test_df)
## 22.5s / Max + 7GB
#peak memory: 21568.29 MiB, increment: 7219.23 MiB
##CPU times: user 26.2 s, sys: 7.28 s, total: 33.5 s
#Wall time: 22.5 s
#%%time
#%memit train_df, test_df = read_feather()
## 1.3s / 7GB (from free -hl)
#peak memory: 13915.43 MiB, increment: 13819.05 MiB
#CPU times: user 1.75 s, sys: 2.18 s, total: 3.93 s
#Wall time: 1.3 s
#train_df.info()
#train_df.head()
##
#test_df.info()
#test_df.head()