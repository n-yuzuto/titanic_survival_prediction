# titanic_survival_prediction(タイタニック号の生存予測モデル構築)

【概要】  
　欠損値がかなり多いタイタニック号の乗客データを見て、訓練データとテストデータに分けて、ある乗客が生存したかどうかを当てるモデルを作りました。データの型を見て、欠損値を補完し、利用する属性などを判断し、学習しやすいデータの型に変換して、学習モデルに入れ、学習させました。今回は、訓練データを70％、テストデータを30％とし、K-fold交差検証を行っています。  

学習の手法としては、以下の4つのメリットから、ランダムフォレストを選択しました。  
①予測精度が高い.  
②特徴量の重要度が評価できる.  
③オーバーフィット（過学習）が起きにくい.  
④複数のツリーの並列処理が可能.  

ソースコードでprintしている部分は、DBがうまく作れているかなどで確認用に書いています。最後の結果のみ出力する場合は、print文を外してください。
