
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor


df_train = pd.read_csv("train_c.csv",parse_dates=["Crash Date/Time"])
df_test = pd.read_csv("test_c.csv",parse_dates=["Crash Date/Time"])

df_train.nunique()

df_train.info()

# Generating few numerical features #
df_train['year']=df_train["Crash Date/Time"].apply(lambda t: t.year)
df_train['month']=df_train["Crash Date/Time"].apply(lambda t: t.month)
df_train['weekday'] = df_train["Crash Date/Time"].apply(lambda t: t.weekday())
df_train['hour']=df_train["Crash Date/Time"].apply(lambda t: t.hour)
df_train['minute']=df_train["Crash Date/Time"].apply(lambda t: t.minute)

df_test['year']=df_test["Crash Date/Time"].apply(lambda t: t.year)
df_test['month']=df_test["Crash Date/Time"].apply(lambda t: t.month)
df_test['weekday'] = df_test["Crash Date/Time"].apply(lambda t: t.weekday())
df_test['hour']=df_test["Crash Date/Time"].apply(lambda t: t.hour)
df_test['minute']=df_test["Crash Date/Time"].apply(lambda t: t.minute)

print(df_train[['weekday','Fault']])


'''df_train = pd.concat([df_train,df_train['Location'].str.split(' ',expand=True)],axis=1)
idx=[]
for i in range(51490):
    if df_train[1][i]==None:
        idx.append(i)
df_new = df_train.iloc[idx]
df_train = df_train.drop(idx)
df_new = df_new.drop(df_new.columns[-1],axis=1)
df_new = df_new.drop(df_new.columns[-1],axis=1)
df_new = pd.concat([df_new,df_new['Location'].str.split(',',expand=True)],axis=1)
df_train = pd.concat([df_train,df_new],ignore_index=True)
df_test = pd.concat([df_test,df_test['Location'].str.split(' ',expand=True)],axis=1)
idx1=[]
for i in range(77235):
    if df_test[1][i]==None:
        idx1.append(i)
        
df_new_test = df_test.iloc[idx1]
df_test = df_test.drop(idx1)
df_new_test = df_new_test.drop(df_new_test.columns[-1],axis=1)
df_new_test = df_new_test.drop(df_new_test.columns[-1],axis=1)
df_new_test = pd.concat([df_new_test,df_new_test['Location'].str.split(',',expand=True)],axis=1)
df_test = pd.concat([df_test,df_new_test],ignore_index=True)
df_train = df_train.rename({
    0:'lat',
    1:'long'
},axis=1)
df_test = df_test.rename({
    0:'lat',
    1:'long'
},axis=1)

df_train['lat'] = pd.to_numeric(df_train['lat'],downcast='float')
df_train['long'] = pd.to_numeric(df_train['long'],downcast='float')
df_test['lat'] = pd.to_numeric(df_test['lat'],downcast='float')
df_test['long'] = pd.to_numeric(df_test['long'],downcast='float')'''

# The required features #
features = [ 'Route Type','Injury Severity',
       'Vehicle Continuing Dir','Vehicle Going Dir',
               'Cross-Street Type','Collision Type','ACRS Report Type',
              'Weather','Surface Condition','Light','Traffic Control',
              'Driver Substance Abuse',
              'Vehicle First Impact Location','Vehicle Second Impact Location',
              'Vehicle Body Type','Vehicle Damage Extent',
              'Vehicle Movement','Driverless Vehicle','Parked Vehicle',
            'Latitude','Longitude','Vehicle Year','Speed Limit','year',
            'month','hour','minute','Fault'
            ]
df_train = df_train[features]
# Categorical features #
cat = ['Route Type','Injury Severity',
       'Vehicle Continuing Dir','Vehicle Going Dir',
               'Cross-Street Type','Collision Type','ACRS Report Type',
              'Weather','Surface Condition','Light','Traffic Control',
              'Driver Substance Abuse',
              'Vehicle First Impact Location','Vehicle Second Impact Location',
              'Vehicle Body Type','Vehicle Damage Extent',
              'Vehicle Movement','Driverless Vehicle','Parked Vehicle']
df_train[cat]=df_train[cat].fillna(df_train.mode().iloc[0])
df_test[cat]=df_test[cat].fillna(df_test.mode().iloc[0])

# Converting categorical features to numerical using OneHotEncoding #
en = OneHotEncoder(handle_unknown='ignore')
for i in cat:
    en_df = []
    en_df = pd.DataFrame(en.fit_transform(df_train[[i]]).toarray())
    en_df.columns = en.get_feature_names([i]) 
    df_train = df_train.drop([i],axis=1)
    df_train = df_train.join(en_df)
for j in cat:
    en_df = []
    en_df = pd.DataFrame(en.fit_transform(df_test[[j]]).toarray())
    en_df.columns = en.get_feature_names([j]) 
    df_test = df_test.drop([j],axis=1)
    df_test = df_test.join(en_df)

feature_array = df_train.columns  # Array of the features #
feature_array_x = feature_array.drop(['Fault'])

y = df_train['Fault']
x = df_train[feature_array_x]
x_test = df_test[feature_array_x]

# To increase the training data , we used SMOTE #
from collections import Counter
sm = SMOTE(sampling_strategy='auto', k_neighbors=1, random_state=0)
X_res, y_res = sm.fit_resample(x, y)
print(Counter(y_res))
df = pd.DataFrame(X_res)
df.columns = feature_array_x
df = df.join(pd.DataFrame(y_res))
X = df[feature_array_x]
df.rename(columns={0:'Fault'},inplace=True)
df.columns
Y = df['Fault']
#Counter(Y)

# Spliting the data for validation #
x_train,x_vldn,y_train,y_vldn = train_test_split(X,Y,test_size=0.2,random_state=0)

#y_train
print(Y)

#mdl = KNeighborsClassifier(n_neighbors=10)
#mdl = GradientBoostingClassifier(max_depth=10,learning_rate=0.13,n_estimators=200,random_state=0)


## Used hyperopt to find the optimal HyperParameters ##
import hyperopt as hp
mdl = XGBRegressor(objective= 'binary:logistic', n_jobs=-1, random_state=42,
                               n_estimators=5000, max_depth=5, learning_rate=0.01, 
                               subsample=0.8, colsample_bytree=0.8,scale_pos_weight=1,gamma=0,seed=200,min_child_weight=6)
mdl.fit(x_train,y_train) # Training the model using only training part of data #
y_pred = mdl.predict(x_vldn)
#y_pred_fi = pd.DataFrame(y_pred,columns=['Fault'])
accuracy = accuracy_score(y_vldn,np.round(y_pred))
print(accuracy)

# Accuracy for the validation set #
y_pred = np.round(y_pred)
accuracy = accuracy_score(y_vldn,y_pred)
print(accuracy)

# Training the model using the entire data #
mdl_fi = XGBRegressor(objective= 'binary:logistic', n_jobs=-1, random_state=42,
                               n_estimators=5000, max_depth=5, learning_rate=0.01, 
                               subsample=0.8, colsample_bytree=0.8,scale_pos_weight=1,gamma=0,seed=200,min_child_weight=6)
mdl_fi.fit(X,Y)
y_pred_final = mdl_fi.predict(x_test)
y_pred_final = np.round(y_pred_final).astype(int)
y_pred_final = pd.DataFrame(y_pred_final,columns=['Fault'])
final_label = pd.DataFrame(df_test['Id'])
final_csv = pd.concat([final_label,y_pred_final], axis=1, join='inner')

final_csv.to_csv("submission.csv",index = False)