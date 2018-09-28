# 直感Deep Learningを読んで面白いことする会 #2 (エムスリー 福林)

## 概要

それっぽい固有名詞を生成したい

## 資料

https://docs.google.com/presentation/d/1gH54QGfOJV6467om_E7fe_N5WEHLSEEMBi5IfOB-dwQ/edit?usp=sharing


### 下準備

python3.6.6で動作確認しています。

```bash
% pip install -r requirements.txt
```

### 学習データの取得

```bash

% python src/retrieve_dbpedia.py src/staions.sparql data/stations.txt
% python src/retrieve_dbpedia.py src/idol-groups.sparql data/idol-groups.txt
% python src/retrieve_dbpedia.py src/idol-names.sparql data/idol-names.txt

```

### 学習と生成

```bash
% python src/rnn.py --hidden 128 --seq 2 --iter 50 data/stations.txt
% python src/rnn.py --hidden 128 --seq 2 --iter 50 data/idol-groups.txt
% python src/rnn.py --hidden 128 --seq 2 --iter 50 data/idol-names.txt
% python src/rnn.py --hidden 128 --seq 2 --iter 200 data/tif2018.txt
```
