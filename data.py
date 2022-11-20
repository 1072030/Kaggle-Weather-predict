import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
from sklearn.neural_network import MLPClassifier
#-------------------------------------讀取檔案
df_X = pd.read_csv('train.csv',index_col='Attribute17')
df_Y = pd.read_csv('train.csv',usecols=['Attribute17'])
df_X['Attribute1'] = pd.to_datetime(df_X['Attribute1'])
df_X = df_X.drop('Attribute1',axis=1) #將日期去除
#-------------------------------------SimpleImputer 處理空值 Strategy 使用最常出現的值(因為有文字)
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(df_X.values)
Y = imputer.fit_transform(df_Y.values)
#-------------------------------------處理文字
lable_1 = LabelEncoder()
X[:,6] = lable_1.fit_transform(X[:,6]) 
X[:,8] = lable_1.fit_transform(X[:,8])
X[:,-1] = lable_1.fit_transform(X[:,-1])
Y = lable_1.fit_transform(Y)
#-------------------------------------資料測試分類
[X_train, X_test, Y_train, Y_test] = train_test_split(X,Y,test_size=0.2,random_state=60,shuffle=True,)
#-------------------------------------Model
classifier = MLPClassifier(activation='relu',solver='adam',alpha=0.0001)
classifier.fit(X_train,Y_train)
#-------------------------------------分數
print("訓練分數")
print(classifier.score(X_train,Y_train)) #訓練分數
Y_pred = classifier.predict(X_test)
joblib.dump(classifier, 'model.pkl') # 將model存入檔案內
print("測試分數")
print(accuracy_score(Y_test,Y_pred)) #測試分數
#-------------------------------------用test.csv 來進行測試
df_test = pd.read_csv('test.csv') #
df_test = df_test.drop('Attribute1',axis=1)
df_answer = pd.read_csv('ex_submit.csv',usecols=['ans'])
#-------------------------------------將空值補滿
Z = imputer.fit_transform(df_test.values)
#-------------------------------------處理文字
lable_2 = LabelEncoder()
Z[:,6] = lable_2.fit_transform(Z[:,6]) 
Z[:,8] = lable_2.fit_transform(Z[:,8])
Z[:,-1] = lable_2.fit_transform(Z[:,-1])
Z_model = joblib.load('model.pkl') #將前面的model用來預測
Z_pred = Z_model.predict(Z)
#-------------------------------------把預測csv檔案匯出
submission = pd.read_csv('ex_submit.csv')
submission["ans"] = Z_pred
submission.to_csv('ex_submit_result.csv',index=False)
#-------------------------------------測試csv檔案分數
print("用test.csv的測試分數")
print(accuracy_score(df_answer.values,Z_pred))