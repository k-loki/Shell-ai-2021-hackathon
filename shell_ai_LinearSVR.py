import warnings

import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import LinearSVR

warnings.filterwarnings('ignore','warnings ignored')
random_seed=42

path = r'data\train.csv'
# load data
df = pd.read_csv(path)

# preprocessing
def preprocess(df,frac,mode = 'train'):
  if mode=='train':
    # fill missing data
    df['Total Cloud Cover [%]'].replace(-7999,np.nan,inplace = True)
    df['Total Cloud Cover [%]'].replace(-6999,np.nan,inplace = True)
    df['Total Cloud Cover [%]'].interpolate(limit = 10,limit_direction = 'both',inplace = True)  

    #  create targets
    df['t_30'] = df.groupby('DATE (MM/DD)')['Total Cloud Cover [%]'].shift(periods = -30,fill_value = -1)
    df['t_60'] = df.groupby('DATE (MM/DD)')['Total Cloud Cover [%]'].shift(periods = -60,fill_value = -1)
    df['t_90'] = df.groupby('DATE (MM/DD)')['Total Cloud Cover [%]'].shift(periods = -90,fill_value = -1)
    df['t_120'] = df.groupby('DATE (MM/DD)')['Total Cloud Cover [%]'].shift(periods = -120,fill_value = -1)

    cond = (df['Total Cloud Cover [%]'] == -1)
    req_samples = df[cond].sample(frac = frac,random_state = random_seed)
    not_req_samples = df[cond].drop(req_samples.index)
    df.drop(not_req_samples.index,inplace=True)

    # drop unwanted features
    df.drop([
            'DATE (MM/DD)',
            'MST',
            'Direct sNIP [W/m^2]',                # this feature is highly correlated with cmp22
            'Tower Wet Bulb Temp [deg C]',        # highly correlated with other temperature readings
            'Tower Dew Point Temp [deg C]',
            'Snow Depth [cm]',
            'Moisture',
            'Albedo (CMP11)',
            'Precipitation (Accumulated) [mm]',
            'Azimuth Angle [degrees]'
    ],axis =1,inplace = True)

    return df
  if mode == 'test':
    df['Total Cloud Cover [%]'].replace(-7999,np.nan,inplace = True)
    df['Total Cloud Cover [%]'].replace(-6999,np.nan,inplace = True)
    df['Total Cloud Cover [%]'].interpolate(limit = 10,limit_direction = 'both',inplace = True)  

    df.drop(columns={
      'Time [Mins]',
      'Direct sNIP [W/m^2]',                # this feature is highly correlated with cmp22
      'Tower Wet Bulb Temp [deg C]',        # highly correlated with other temperature readings
      'Tower Dew Point Temp [deg C]',
      'Snow Depth [cm]',
      'Moisture',
      'Albedo (CMP11)',
      'Precipitation (Accumulated) [mm]',
      'Azimuth Angle [degrees]' 
    },inplace = True)
    return df.iloc[-1,]

df = preprocess(df,0.1,mode='train')

def split(x,y,train_size=0.50):
  return train_test_split(x,y,train_size=train_size,random_state=random_seed)

X_train_30,X_test_30,Y_train_30,Y_test_30 = split(df.iloc[:,:-4].values,df['t_30'].values)
X_train_60,X_test_60,Y_train_60,Y_test_60 = split(df.iloc[:,:-3].values,df['t_60'].values)
X_train_90,X_test_90,Y_train_90,Y_test_90 = split(df.iloc[:,:-2].values,df['t_90'].values)
X_train_120,X_test_120,Y_train_120,Y_test_120 = split(df.iloc[:,:-1].values,df['t_120'].values)

# scaling and transforming train and test data
def scale_and_transform(data,scaler=None,power=None,mode='train'):
  if mode=='train':
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    power = PowerTransformer()
    data = power.fit_transform(data)
    return data,scaler,power
  if mode == 'test' and scaler is not None:
    data = scaler.transform(data)
    data = power.transform(data)
    return data

X_train_30,scaler_30,pow_30 = scale_and_transform(X_train_30,mode='train')
X_test_30 = scale_and_transform(X_test_30,scaler_30,pow_30,mode='test')

X_train_60,scaler_60,pow_60 = scale_and_transform(X_train_60,mode='train')
X_test_60 = scale_and_transform(X_test_60,scaler_60,pow_60,mode='test')

X_train_90,scaler_90,pow_90 = scale_and_transform(X_train_90,mode='train')
X_test_90 = scale_and_transform(X_test_90,scaler_90,pow_90,mode='test')

X_train_120,scaler_120,pow_120 = scale_and_transform(X_train_120,mode='train')
X_test_120 = scale_and_transform(X_test_120,scaler_120,pow_120,mode='test')

model_30 = LinearSVR(verbose=1,max_iter=100000)
model_30.fit(X_train_30,Y_train_30)
preds_30  = model_30.predict(X_test_30)
score_30 = mae(Y_test_30,preds_30)
print(f"model_30 mae score: {score_30}")
# dump(model_30,'new_LinearSVRmodel_30.joblib')


model_60 = LinearSVR(verbose=1,max_iter=100000)
model_60.fit(X_train_60,Y_train_60)
preds_60  = model_60.predict(X_test_60)
score_60 = mae(Y_test_60,preds_60)
print(f"model_60 mae score: {score_60}")
# dump(model_60,'new_LinearSVRmodel_60.joblib')


model_90 = LinearSVR(verbose=1,max_iter=100000)
model_90.fit(X_train_90,Y_train_90)
preds_90  = model_90.predict(X_test_90)
score_90 = mae(Y_test_90,preds_90)
print(f"model_90 mae score: {score_90}")
# dump(model_90,'new_LinearSVRmodel_90.joblib')


model_120 = LinearSVR(verbose=1,max_iter=100000)
model_120.fit(X_train_120,Y_train_120)
preds_120  = model_120.predict(X_test_120)
score_120 = mae(Y_test_120,preds_120)
print(f"model_120 mae score: {score_120}")
# dump(model_120,'new_LineraSVRmodel_120.joblib')


test_df = pd.read_csv(r'data\unused_df.csv')
test_df = preprocess(test_df,0.2,mode='train')

X_30,Y_30 = test_df.iloc[:,:-4].values,test_df['t_30'].values
c_30 = cross_val_score(model_30,X_30,Y_30,scoring='neg_mean_absolute_error',verbose=1,n_jobs=-1)
print(f"c_30:{c_30} ,mean---------> {c_30.mean()}")

X_60,Y_60 = test_df.iloc[:,:-3].values,test_df['t_60'].values
c_60 = cross_val_score(model_60,X_60,Y_60,scoring='neg_mean_absolute_error',verbose=1,n_jobs=-1)
print(f"c_60:{c_60 } ,mean---------> { c_60.mean()}")

X_90,Y_90 = test_df.iloc[:,:-2].values,test_df['t_90'].values
c_90 = cross_val_score(model_90,X_90,Y_90,scoring='neg_mean_absolute_error',verbose=1,n_jobs=-1)
print(f"c_90:{c_90 } ,mean---------> { c_90.mean()}")

X_120,Y_120 = test_df.iloc[:,:-1].values,test_df['t_120'].values
c_120 = cross_val_score(model_120,X_120,Y_120,scoring='neg_mean_absolute_error',verbose=1,n_jobs=-1)
print(f"c_120:{c_120} ,mean---------> {c_120.mean()}")

