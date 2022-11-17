import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.utils.validation import column_or_1d

df_X = pd.read_csv('train.csv',index_col='Attribute17')
df_Y = pd.read_csv('train.csv',usecols=['Attribute17'])
df_X['Attribute1'] = pd.to_datetime(df_X['Attribute1'])
print(df_X)
df_X = df_X.drop('Attribute1',axis=1)
#------------------SimpleImputer 處理空值 Strategy 使用最常出現的值(因為有文字)
imputer = SimpleImputer(missing_values=np.nan,strategy='most_frequent')


X = imputer.fit_transform(df_X.values)
Y = imputer.fit_transform(df_Y.values)

#----------------------處理文字
lable_1 = LabelEncoder()
X[:,6] = lable_1.fit_transform(X[:,6]) 
lable_2 = LabelEncoder()
X[:,8] = lable_2.fit_transform(X[:,8])
lable_3 = LabelEncoder()
X[:,-1] = lable_3.fit_transform(X[:,-1])

lable_Y = LabelEncoder()
Y = lable_Y.fit_transform(Y)
#---------------------------------資料前處理
sc = StandardScaler()
X=sc.fit_transform(X)
#--------------------------------
[X_train, X_test, Y_train, Y_test] = train_test_split(X,Y,test_size=806,random_state=0)
classifier = DecisionTreeClassifier(criterion='entropy',min_samples_split=2,min_samples_leaf=1,min_weight_fraction_leaf=0.5)
classifier.fit(X_train,Y_train)
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
Z = imputer.fit_transform(df_test.values)
# ans = imputer.fit_transform(df_answer.values)
lable_4 = LabelEncoder()
Z[:,6] = lable_4.fit_transform(Z[:,6]) 
lable_5 = LabelEncoder()
Z[:,8] = lable_5.fit_transform(Z[:,8])
lable_6 = LabelEncoder()
Z[:,-1] = lable_6.fit_transform(Z[:,-1])

Z = sc.fit_transform(Z)
Z_model = joblib.load('model.pkl')
Z_pred = Z_model.predict(Z)
Z_pred = Z_pred.reshape(-1,1)

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


print(accuracy_score(df_answer.values,Z_pred))
# print(Z_pred)
# print(accuracy_score(Y_test,Y_pred))




