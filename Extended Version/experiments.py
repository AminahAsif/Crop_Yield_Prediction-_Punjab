import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, json
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

SRC='/home/claude/w/data/data/processed/final_ml_dataset_cv.csv'
raw=pd.read_csv(SRC)
# ---- deduplicate the nine-fold replication: one row per crop-year, mean temperature
g=raw.groupby(['crop','year'])
d=g.agg(yield_ton_ha=('yield_ton_ha','first'),pesticides_tonnes=('pesticides_tonnes','first'),
        avg_temp=('avg_temp','mean'),avg_temp_sd=('avg_temp','std'),n_rows=('avg_temp','size'),
        N=('N','first'),P=('P','first'),K=('K','first'),temperature=('temperature','first'),humidity=('humidity','first'),
        ph=('ph','first'),NPK_total=('NPK_total','first'),health_score=('health_score','first')).reset_index()
d['crop_code']=(d.crop=='wheat').astype(int)
d.to_csv('/home/claude/w/paper/code/crop_year_dataset.csv',index=False)
print(d.shape, d.n_rows.unique())
A=['pesticides_tonnes','avg_temp','N','P','K','temperature','humidity','ph','NPK_total']
B=A+['health_score']; C=A+['year']

def mk(name):
    if name=='RF': return RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42,n_jobs=1)
    if name=='XGB': return XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42,verbosity=0,n_jobs=1)
    if name=='SVR': return make_pipeline(StandardScaler(),SVR(kernel='rbf',C=100,gamma=0.1,epsilon=0.1))
    if name=='Ridge': return make_pipeline(StandardScaler(),Ridge(alpha=1.0))
    if name=='Stack':
        return StackingRegressor(estimators=[('rf',mk('RF')),('xgb',mk('XGB')),('svr',mk('SVR'))],final_estimator=Ridge(alpha=1.0),cv=5)
def fit_predict(name,Xtr,ytr,Xte,tr,te):
    if name=='CropMean':
        m=tr.groupby('crop').yield_ton_ha.mean(); return te.crop.map(m).values
    if name=='CropTrend':
        out=np.zeros(len(te))
        for c in te.crop.unique():
            t=tr[tr.crop==c]; lr=LinearRegression().fit(t[['year']],t.yield_ton_ha); idx=(te.crop==c).values
            out[idx]=lr.predict(te.loc[idx,['year']])
        return out
    m=mk(name); m.fit(Xtr,ytr); return m.predict(Xte)

def splits(protocol):
    yrs=np.sort(d.year.unique())
    if protocol=='LOYO':
        for y in yrs: yield np.where(d.year!=y)[0], np.where(d.year==y)[0]
    elif protocol=='Forward':
        for y in yrs[yrs>=1998]: yield np.where(d.year<y)[0], np.where(d.year==y)[0]
    elif protocol=='Random':
        rkf=RepeatedKFold(n_splits=5,n_repeats=5,random_state=42)
        for tr,te in rkf.split(d): yield tr,te

def evaluate(name,feats,protocol):
    if protocol=='Random':
        # collect per-repeat pooled metrics
        preds={}; 
        rk=list(splits('Random')); res=[]
        for r in range(5):
            yt=[];yp=[]
            for tr,te in rk[r*5:(r+1)*5]:
                p=fit_predict(name,d.loc[tr,feats],d.yield_ton_ha.iloc[tr],d.loc[te,feats],d.iloc[tr],d.iloc[te]); yt+=list(d.yield_ton_ha.iloc[te]); yp+=list(p)
            yt=np.array(yt);yp=np.array(yp)
            res.append((np.sqrt(mean_squared_error(yt,yp)),mean_absolute_error(yt,yp),r2_score(yt,yp)))
        res=np.array(res); return dict(rmse=res[:,0].mean(),mae=res[:,1].mean(),r2=res[:,2].mean(),rmse_sd=res[:,0].std()),None
    yt=[];yp=[];yr=[];cr=[]
    for tr,te in splits(protocol):
        p=fit_predict(name,d.loc[tr,feats],d.yield_ton_ha.iloc[tr],d.loc[te,feats],d.iloc[tr],d.iloc[te])
        yt+=list(d.yield_ton_ha.iloc[te]);yp+=list(p);yr+=list(d.year.iloc[te]);cr+=list(d.crop.iloc[te])
    yt=np.array(yt);yp=np.array(yp);yr=np.array(yr)
    # bootstrap over years
    rng=np.random.default_rng(0); ys=np.unique(yr); bs=[]
    for _ in range(2000):
        pick=rng.choice(ys,len(ys)); idx=np.concatenate([np.where(yr==y)[0] for y in pick])
        bs.append(np.sqrt(np.mean((yt[idx]-yp[idx])**2)))
    out=dict(rmse=float(np.sqrt(mean_squared_error(yt,yp))),mae=float(mean_absolute_error(yt,yp)),r2=float(r2_score(yt,yp)),
             ci_lo=float(np.percentile(bs,2.5)),ci_hi=float(np.percentile(bs,97.5)))
    return out,pd.DataFrame(dict(year=yr,crop=cr,y=yt,p=yp))

rows=[];store={}
configs=[('CropMean',A,'-'),('CropTrend',A,'-'),('Ridge',A,'A'),('RF',A,'A'),('XGB',A,'A'),('SVR',A,'A'),('Stack',A,'A'),
         ('RF',B,'B'),('XGB',B,'B'),('Stack',B,'B'),('RF',C,'C'),('XGB',C,'C'),('SVR',C,'C'),('Stack',C,'C'),('Ridge',C,'C')]
import sys, os
PROTS=sys.argv[1:]
for prot in PROTS:
    for name,feats,fs in configs:
        r,pr=evaluate(name,feats,prot); r.update(model=name,features=fs,protocol=prot); rows.append(r)
        if pr is not None: pr.to_csv(f'/home/claude/w/paper/code/preds_{prot}_{name}_{fs}.csv',index=False)
        pd.DataFrame(rows).to_csv('/home/claude/w/paper/code/protocol_results_partial.csv',index=False)
        print(prot,name,fs,{k:round(v,4) for k,v in r.items() if isinstance(v,float)},flush=True)
pd.DataFrame(rows).to_csv('/home/claude/w/paper/code/protocol_results.csv',index=False)
