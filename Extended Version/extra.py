import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from xgboost import XGBRegressor
raw=pd.read_csv('/home/claude/w/data/data/processed/final_ml_dataset_cv.csv')
A=['pesticides_tonnes','avg_temp','N','P','K','temperature','humidity','ph','NPK_total']
mk={'RF':lambda:RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42,n_jobs=1),
    'XGB':lambda:XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42,verbosity=0,n_jobs=1),
    'SVR':lambda:make_pipeline(StandardScaler(),SVR(C=100,gamma=0.1,epsilon=0.1))}
X=raw[A]; y=raw.yield_ton_ha.values; yrs=raw.year.values
out=[]
for n,f in mk.items():
    # row-level random 5-fold (one repeat, shuffled)
    p=np.zeros(len(y))
    for tr,te in KFold(5,shuffle=True,random_state=42).split(X): p[te]=f().fit(X.iloc[tr],y[tr]).predict(X.iloc[te])
    r1=(np.sqrt(mean_squared_error(y,p)),r2_score(y,p))
    p=np.zeros(len(y))
    for yr in np.unique(yrs):
        te=yrs==yr; p[te]=f().fit(X[~te],y[~te]).predict(X[te])
    r2=(np.sqrt(mean_squared_error(y,p)),r2_score(y,p))
    out.append(dict(model=n,random5_rmse=r1[0],random5_r2=r1[1],loyo_rmse=r2[0],loyo_r2=r2[1])); print(out[-1],flush=True)
pd.DataFrame(out).to_csv('extra414.csv',index=False)
d=pd.read_csv('crop_year_dataset.csv')
print('corr pesticides~year',d[['pesticides_tonnes','year']].corr().iloc[0,1].round(3))
for c in ['maize','wheat']:
    s=d[d.crop==c]; print(c,'yield~year',round(np.corrcoef(s.year,s.yield_ton_ha)[0,1],3),'yield~pest',round(np.corrcoef(s.pesticides_tonnes,s.yield_ton_ha)[0,1],3),'yield~temp',round(np.corrcoef(s.avg_temp,s.yield_ton_ha)[0,1],3),'pest~year',round(np.corrcoef(s.year,s.pesticides_tonnes)[0,1],3))
print('avg_temp within-year sd mean',d.avg_temp_sd.mean().round(3),'between-year sd',d.groupby('year').avg_temp.mean().std().round(3))
import scipy.stats as st
for k,n in [(771,772)]:
    z=1.96; ph=k/n; c=(ph+z*z/(2*n))/(1+z*z/n); h=z*np.sqrt(ph*(1-ph)/n+z*z/(4*n*n))/(1+z*z/n); print('wilson',c-h,c+h)
print(d.groupby('crop').yield_ton_ha.agg(['mean','std','min','max']).round(3))
