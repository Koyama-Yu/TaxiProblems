# TaxiProblems リポジトリ

Taxi問題のシミュレーションを行うプログラム集
フォルダの管理が面倒だったので，各フォルダにほぼ同じ構造でファイルが配置されています(汚くて申し訳ない)

## 各ディレクトリ

- `Draw`: 実験結果(最適行動選択率)の描画用
- `TaxiProblem`: ε-greedy法、中でさらにε固定とε更新のもので分かれています
- `TaxiProblem_softmax`: Softmax法(パラメータチューニングうまくいかず)
- `TaxiProblem_SpeesyQ`: SpeedyQを実装、比較しようとしたが結局使っていない
- `TaxiProblemwithLA`: (βタイプ)学習オートマトンを用いた手法、内部にベイズ推定器やβタイプ学習オートマトンを実装したソースコードを含みます

## 使い方
基本的にはコード読めばわかりますが
1. Const/constant.pyのパラメータ値を編集
2. taxiproblem.py(mainファイル)を実行
3. glaphs,probabilitiesフォルダに結果が出てくるのでそれを参照する(パラメータを変えるとファイル名変わる)
4. 各手法との比較などしたい場合はprobabilitiesのファイルをDrawに持ってきてそれを使って描画
すればできます

何かあれば連絡ください
