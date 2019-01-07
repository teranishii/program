## Attention Summary
生成要約プログラムです．（※ chainer で書かれてます）  
`train` の中に訓練データ `test` の中にテストデータが入ってます．(10 件のみ)  
基本的には`attention.py`を実行するだけで動くはず...
### 実行環境
```
GPU : GeForce 1080  
メモリ : 8GB
```
### 実行例  
CPU  
`python attention.py`  
GPU  
`python attention.py --gpu=1`

### ライブラリ
必要そうなライブラリのバージョンです．
```
chainer 4.4.0  
cupy 4.4.1  
h5py 2.8.0  
numpy 1.15.1
```
