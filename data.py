import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler 
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.utils.validation import column_or_1d

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

df_X = pd.read_csv('train.csv',index_col='Attribute17')
df_Y = pd.read_csv('train.csv',usecols=['Attribute17'])
df_X['Attribute1'] = pd.to_datetime(df_X['Attribute1'])
# print(df_X)
df_X = df_X.drop('Attribute1',axis=1)
#------------------SimpleImputer 處理空值 Strategy 使用最常出現的值(因為有文字)
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')
X = imputer.fit_transform(df_X.values)
Y = imputer.fit_transform(df_Y.values)
# print(X)

# print(Y)
# # print("fillna-------------------------------------------------------")
# df_X["Attribute2"]= df_X["Attribute2"].fillna(df_X["Attribute2"].mean())
# df_X["Attribute3"]= df_X["Attribute3"].fillna(df_X["Attribute3"].mean())
# df_X["Attribute4"]= df_X["Attribute4"].fillna(df_X["Attribute4"].mean())
# df_X["Attribute5"]= df_X["Attribute5"].fillna(df_X["Attribute5"].mean())
# df_X["Attribute6"]= df_X["Attribute6"].fillna(df_X["Attribute6"].mean())
# df_X["Attribute7"]= df_X["Attribute7"].fillna(df_X["Attribute7"].mean())
# df_X["Attribute9"]= df_X["Attribute9"].fillna(df_X["Attribute9"].mean())
# df_X["Attribute11"]= df_X["Attribute11"].fillna(df_X["Attribute11"].mean())
# df_X["Attribute12"]= df_X["Attribute12"].fillna(df_X["Attribute12"].mean())
# df_X["Attribute13"]= df_X["Attribute13"].fillna(df_X["Attribute13"].mean())
# df_X["Attribute14"]= df_X["Attribute14"].fillna(df_X["Attribute14"].mean())
# df_X["Attribute15"]= df_X["Attribute15"].fillna(df_X["Attribute15"].mean())
# df_X['Attribute8']=df_X['Attribute8'].fillna(df_X['Attribute8'].mode()[0])
# df_X['Attribute10'] = df_X['Attribute10'].fillna(df_X['Attribute10'].mode()[0])
# df_X['Attribute16'] = df_X['Attribute16'].fillna(df_X['Attribute16'].mode()[0])
# df_Y['Attribute17'] = df_Y['Attribute17'].fillna(df_Y['Attribute17'].mode()[0])
# X = df_X.values
# Y = df_Y.values
# print(X)
#----------------------處理文字
lable_1 = LabelEncoder()
X[:,6] = lable_1.fit_transform(X[:,6]) 
X[:,8] = lable_1.fit_transform(X[:,8])
X[:,-1] = lable_1.fit_transform(X[:,-1])
Y = lable_1.fit_transform(Y)
#---------------------------------資料前處理
sc = StandardScaler()
X=sc.fit_transform(X)
#--------------------------------
[X_train, X_test, Y_train, Y_test] = train_test_split(X,Y,test_size=0.2,random_state=50)
classifier = DecisionTreeClassifier(criterion='gini',min_samples_split=3,min_samples_leaf=5,splitter="random",random_state=50)
# criterion 選擇 gini 因為我們數據較大

# classifier = RandomForestClassifier(n_estimators=150,criterion='gini',min_samples_split=3,min_samples_leaf=5,random_state=50,oob_score=True)
# classifier = GradientBoostingClassifier(n_estimators=100,random_state=50)
# classifier = LogisticRegression()
classifier.fit(X_train,Y_train)

print("classifier.score")
print(classifier.score(X_train,Y_train))
Y_pred = classifier.predict(X_test)

joblib.dump(classifier, 'model.pkl')
# submission = pd.read_csv('ex_submit_copy.csv')
# submission.to_csv('ex_submit_copy.csv',index=False)
print("accuracy Y_test Y_pred")
print(accuracy_score(Y_test,Y_pred))
#--------------------------------------------
df_test = pd.read_csv('test.csv')
df_test = df_test.drop('Attribute1',axis=1)
df_answer = pd.read_csv('ex_submit.csv',usecols=['ans'])
# print(df_answer.values.squeeze())
#---------------------------------------------------
Z = imputer.fit_transform(df_test.values)
# df_test["Attribute2"]= df_test["Attribute2"].fillna(df_test["Attribute2"].mean())
# df_test["Attribute3"]= df_test["Attribute3"].fillna(df_test["Attribute3"].mean())
# df_test["Attribute4"]= df_test["Attribute4"].fillna(df_test["Attribute4"].mean())
# df_test["Attribute5"]= df_test["Attribute5"].fillna(df_test["Attribute5"].mean())
# df_test["Attribute6"]= df_test["Attribute6"].fillna(df_test["Attribute6"].mean())
# df_test["Attribute7"]= df_test["Attribute7"].fillna(df_test["Attribute7"].mean())
# df_test["Attribute9"]= df_test["Attribute9"].fillna(df_test["Attribute9"].mean())
# df_test["Attribute11"]= df_test["Attribute11"].fillna(df_test["Attribute11"].mean())
# df_test["Attribute12"]= df_test["Attribute12"].fillna(df_test["Attribute12"].mean())
# df_test["Attribute13"]= df_test["Attribute13"].fillna(df_test["Attribute13"].mean())
# df_test["Attribute14"]= df_test["Attribute14"].fillna(df_test["Attribute14"].mean())
# df_test["Attribute15"]= df_test["Attribute15"].fillna(df_test["Attribute15"].mean())
# df_test['Attribute8']=df_test['Attribute8'].fillna(df_test['Attribute8'].mode()[0])
# df_test['Attribute10'] = df_test['Attribute10'].fillna(df_test['Attribute10'].mode()[0])
# df_test['Attribute16'] = df_test['Attribute16'].fillna(df_test['Attribute16'].mode()[0])
# Z = df_test.values
#---------------------------------------------------
lable_4 = LabelEncoder()
Z[:,6] = lable_4.fit_transform(Z[:,6]) 
Z[:,8] = lable_4.fit_transform(Z[:,8])
Z[:,-1] = lable_4.fit_transform(Z[:,-1])

Z = sc.fit_transform(Z)
Z_model = joblib.load('model.pkl')
Z_pred = Z_model.predict(Z)
Z_pred = Z_pred.reshape(-1,1)
# print(Z_pred)


# print(ans)
# print(Z_pred)

submission = pd.read_csv('ex_submit.csv')
submission["ans"] = Z_pred
submission.to_csv('ex_submit_copy.csv',index=False)

# Z_model = lable_Y.inverse_transform(Z_model)
# ans = lable_Y.inverse_transform(ans)
# Z_model = Z_model.reshape(-1,1)
# asn = ans.reshape(-1,1)
# print(Z_model)
# print(ans)

print("compare answer")
print(accuracy_score(df_answer.values,Z_pred))
# print(Z_pred)
# print(accuracy_score(Y_test,Y_pred))




