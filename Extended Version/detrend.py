import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
d=pd.read_csv('crop_year_dataset.csv'); d['crop_code']=(d.crop=='wheat').astype(int)
F=['pesticides_tonnes','avg_temp','crop_code']
mk={'Ridge':lambda:make_pipeline(StandardScaler(),Ridge(alpha=1.0)),
    'RF':lambda:RandomForestRegressor(n_estimators=200,max_depth=3,min_samples_leaf=3,random_state=42,n_jobs=1),
    'XGB':lambda:XGBRegressor(n_estimators=100,learning_rate=0.05,max_depth=2,random_state=42,verbosity=0,n_jobs=1),
    'SVR':lambda:make_pipeline(StandardScaler(),SVR(C=1,gamma='scale',epsilon=0.05))}
def trend_pred(tr,te):
    out=np.zeros(len(te)); fit=np.zeros(len(tr))
    for c in d.crop.unique():
        t=tr[tr.crop==c]; lr=LinearRegression().fit(t[['year']],t.yield_ton_ha)
        out[(te.crop==c).values]=lr.predict(te.loc[te.crop==c,['year']]); fit[(tr.crop==c).values]=lr.predict(t[['year']])
    return out,fit
def run(protocol):
    yrs=np.sort(d.year.unique()); res={k:[] for k in ['Trend']+list(mk)}; yy=[];yr=[]
    tests=[y for y in yrs if (protocol=='LOYO' or y>=1998)]
    for y in tests:
        tr=d[d.year!=y] if protocol=='LOYO' else d[d.year<y]; te=d[d.year==y]
        tp,fit=trend_pred(tr,te); res['Trend']+=list(tp); yy+=list(te.yield_ton_ha); yr+=[y]*len(te)
        resid=tr.yield_ton_ha.values-fit
        for k,f in mk.items(): res[k]+=list(tp+f().fit(tr[F],resid).predict(te[F]))
    yy=np.array(yy); yr=np.array(yr); rng=np.random.default_rng(0); ys=np.unique(yr); out=[]
    for k,p in res.items():
        p=np.array(p); rm=np.sqrt(np.mean((yy-p)**2)); bs=[]
        for _ in range(2000):
            pick=rng.choice(ys,len(ys)); idx=np.concatenate([np.where(yr==a)[0] for a in pick]); bs.append(np.sqrt(np.mean((yy[idx]-p[idx])**2)))
        out.append((protocol,k,rm,np.percentile(bs,2.5),np.percentile(bs,97.5),1-np.sum((yy-p)**2)/np.sum((yy-yy.mean())**2)))
    return out
rows=run('LOYO')+run('Forward')
r=pd.DataFrame(rows,columns=['protocol','model','rmse','ci_lo','ci_hi','r2']); r.to_csv('detrend_results.csv',index=False); print(r.round(3).to_string())
# paired comparison vs trend: per-year squared error difference
