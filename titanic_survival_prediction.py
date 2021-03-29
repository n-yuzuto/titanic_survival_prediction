#タイタニック号の生存予測 

#pclass:階級,sibsp:兄弟、配偶者の数;,parch:両親、子供の数,ticket:チケット番号
#fare:運賃,cabin:部屋番号,embark:船に乗った場所

import pandas as pd

# データの読み込み
df = pd.read_csv('titanic3.csv')

df_copy = df.copy()
print(df_copy.head())

# ラベルエンコーディング
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder() #ラベルエンコーダのインスタンスを作成

df_copy['sex'] = le.fit_transform(df_copy['sex']) #エンコーディング
df_copy['embarked'] = le.fit_transform(df_copy['embarked'].astype(str))
print(df.head())
#エンコーディングに成功

# 欠損値補完
#欠損値にageの平均値で補完
df_copy['age'] = df_copy['age'].fillna(df_copy['age'].mean()) 
#欠損値にageの平均値で補完
df_copy['fare'] = df_copy['fare'].fillna(df_copy['fare'].mean()) 
print(df_copy.isnull().sum())
#欠損値の補完に成功
#bodyは結果が残ってなさ過ぎたため、補完をやめた

# 不要行の削除
df_copy = df_copy.drop(['name', 'ticket', 'cabin', 'boat', 'body', 'home.dest'],axis=1)
#学習に名前やチケット番号などはいらないと思い、削除
#また、bodyは関係あると思ったが、データが少なすぎたため、利用するのを辞めた
print(df_copy)
#不要列の削除に成功

# ndarray形式への変換
features = df_copy[['pclass','age','sex','fare','embarked']].values
target = df_copy['survived'].values
#機械学習に用いるため、ndarrayに変換した

# 学習データとテストデータに分割
from sklearn.model_selection import  train_test_split
#今回は７割のデータで学習、３割のデータでテストを行う
(features , test_X , target , test_y) = train_test_split(features, target , test_size = 0.3 , random_state = 0)

# 学習
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100,random_state=0) 
# ランダムフォレストのインスタンスを作成

model.fit(features,target) # 学習の実行

# 予測
pred = model.predict(test_X)
#テストデータに関して、予測を行ってみる
#結果
print(pred)

# 予測精度の確認
from sklearn.metrics import accuracy_score
#正解率にて、予測精度を検証
print(accuracy_score(pred,test_y))


# 重要度の表示
importace = model.feature_importances_ 
#ランダムフォレストの学習における各列（特長量）の重要度を確認する
print('Feature Importances:')
for i, feat in enumerate(['pclass','age','sex','fare','embarked']):
    print('\t{0:20s} : {1:>.5f}'.format(feat, importace[i]))
    
# csvで出力
df_pred = pd.DataFrame(pred)
df_pred.to_csv('submission.csv',header=None)