# solafune「衛星画像から空港利用者数を予測」コンペ用コード
コンペに参加するために作成したコードです。全く結果は残せませんでした。（23/118位）
https://solafune.com/#/competitions/ea90cba4-3e01-42df-9516-9ac0d7a44204

## ディレクトリ構成
.
├── data
│   ├── data_aug　　　　　　<--データ拡張結果の保存（画像）
│   │   ├── evaluatemodel
│   │   │   └── images
│   │   ├── testimage
│   │   │   └── images
│   │   └── trainimage
│   │       └── images
│   ├── ori                <--オリジナルデータの格納（画像）
│   │   ├── evaluatemodel
│   │   │   └── images
│   │   ├── testimage
│   │   │   └── images
│   │   └── trainimage
│   │       └── images
│   └── prep　　　　　　    <--前処理済データの保存（npy）
└── src　　　　　　　　　　　<--ソースコードの格納
    └──result              <--実行結果の格納
       ├── 01_stats
       ├── 02_VGG-like
       └── tflog

## 使い方
基本的な設定はsrc/config.iniで設定可能
1. src/0_DataAug.py　を実行してデータ拡張
2. src/1_PREP_images.py　を実行して前処理
3. src/1_PREP_stat-all.py　を実行して前処理
4. src/2_TRAIN_stats.py　を実行して学習・結果の出力
5. src/2_TRAIN_VGG-like.py　を実行して学習・結果の出力

