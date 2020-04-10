
import rain1

%load_ext memory_profiler
%load_ext line_profiler

%%time
%memit train_df, test_df = main()
# 14s / 15GB



oad_ext memory_profiler
%load_ext line_profiler
%%time
%memit train_df, test_df = main()
# 14s / 15GB
HBox(children=(FloatProgress(value=0.0, max=76345.0), HTML(value='')))
HBox(children=(FloatProgress(value=0.0, max=2416.0), HTML(value='')))
peak memory: 14355.44 MiB, increment: 14259.02 MiB
CPU times: user 3.75 s, sys: 7.17 s, total: 10.9 s
Wall time: 14 s
%%time
%memit to_feather(train_df, test_df)
# 22.5s / Max + 7GB
peak memory: 21568.29 MiB, increment: 7219.23 MiB
CPU times: user 26.2 s, sys: 7.28 s, total: 33.5 s
Wall time: 22.5 s
%%time
%memit train_df, test_df = read_feather()
# 1.3s / 7GB (from free -hl)
peak memory: 13915.43 MiB, increment: 13819.05 MiB
CPU times: user 1.75 s, sys: 2.18 s, total: 3.93 s
Wall time: 1.3 s
train_df.info()
train_df.head()

test_df.info()
test_df.head()