import warnings
import numpy as np
import pandas as pd
from joblib import dump, load
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
warnings.filterwarnings('error','warnings ignored')
random_seed=42

train_path = r'data\train.csv'
# load data
df = pd.read_csv(train_path)

# preprocessing
def preprocess(df,mode = 'train'):
  frac = 0.1
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
    
    # selected fts---> []
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
      'Azimuth Angle [degrees]',
      'scenario_set' 
    },inplace = True)
    return df.iloc[-1,]

df = preprocess(df,mode='train')

def split(x,y,train_size=0.50):
  return train_test_split(x,y,train_size=train_size,random_state=random_seed)

X_train_30,X_test_30,Y_train_30,Y_test_30 = split(df.iloc[:,:-4].values,df['t_30'].values)
X_train_60,X_test_60,Y_train_60,Y_test_60 = split(df.iloc[:,:-3].values,df['t_60'].values)
X_train_90,X_test_90,Y_train_90,Y_test_90 = split(df.iloc[:,:-2].values,df['t_90'].values)
X_train_120,X_test_120,Y_train_120,Y_test_120 = split(df.iloc[:,:-1].values,df['t_120'].values)


params = {'alpha': 0.0001,
          'early_stopping': True,
          'epsilon': 0.1,
          'learning_rate': 'optimal',
          'loss': 'huber',
          'max_iter': 10000,
          'n_iter_no_change': 10,
          'penalty': 'l1',
          'random_state': 42
          }

pipeline_30 = Pipeline([
    ('scaler',StandardScaler()),
    ('transformer',PowerTransformer()),
    ('sgd',SGDRegressor(**params))
])

pipeline_60 = Pipeline([
    ('scaler',StandardScaler()),
    ('transformer',PowerTransformer()),
    ('sgd',SGDRegressor(**params))
])

pipeline_90 = Pipeline([
    ('scaler',StandardScaler()),
    ('transformer',PowerTransformer()),
    ('sgd',SGDRegressor(**params))
])

pipeline_120 = Pipeline([
    ('scaler',StandardScaler()),
    ('transformer',PowerTransformer()),
    ('sgd',SGDRegressor(**params))
])

def train_and_predict(model,x_tr,x_ts,y_tr,y_ts):
  model.fit(x_tr,y_tr)
  print(model.score(x_tr,y_tr))
  preds = model.predict(x_ts)
  print(f"mae score: {mae(y_ts,preds)}")
  return preds

print('---------------Training and validating------------')
pred_30 = train_and_predict(pipeline_30,X_train_30,X_test_30,Y_train_30,Y_test_30)
pred_60 = train_and_predict(pipeline_60,X_train_60,X_test_60,Y_train_60,Y_test_60)
pred_90 = train_and_predict(pipeline_90,X_train_90,X_test_90,Y_train_90,Y_test_90)
pred_120 = train_and_predict(pipeline_120,X_train_120,X_test_120,Y_train_120,Y_test_120)

# cross_val_score(pipeline_30,X_test_30,Y_test_30,)

from joblib import dump

dump(pipeline_30,'SGDr_30_finetuned.joblib')
dump(pipeline_60,'SGDr_60_finetuned.joblib')
dump(pipeline_90,'SGDr_90_finetuned.joblib')
dump(pipeline_120,'SGDr_120_finetuned.joblib')

print('-------------Final predictions---------')

wd_test = pd.read_csv(r'data\shell_test.csv')
test = pd.read_csv(r'data\test.csv')

file_count = 0
for set in range(1,301):
  wd = wd_test[wd_test['scenario_set'] == set]
  print(wd)
  # preprocessing test data
  last_sample = preprocess(wd,mode='test')

  # predicting test samples
  pred_30 = pipeline_30.predict(last_sample.values.reshape(1,-1))
  last_sample['t_30'] = pred_30.item()
  pred_60 = pipeline_60.predict(last_sample.values.reshape(1,-1))
  last_sample['t_60'] = pred_60.item()
  pred_90 = pipeline_90.predict(last_sample.values.reshape(1,-1))
  last_sample['t_90'] = pred_90.item()
  pred_120 = pipeline_120.predict(last_sample.values.reshape(1,-1))
  
  # fill in test data using above predictions
  test.iloc[set-1,test.columns.get_indexer(['30_min_horizon'])] = np.round(pred_30.item())
  test.iloc[set-1,test.columns.get_indexer(['60_min_horizon'])] = np.round(pred_60.item())
  test.iloc[set-1,test.columns.get_indexer(['90_min_horizon'])] = np.round(pred_90.item())
  test.iloc[set-1,test.columns.get_indexer(['120_min_horizon'])] = np.round(pred_120.item())

  file_count += 1
  if file_count%30 == 0 :
      print(file_count)

test = test.applymap(int)
test.to_csv('SGDr_finetuned_preds.csv',index=False)

print(test.describe())
print(test)
