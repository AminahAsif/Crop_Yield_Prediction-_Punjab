import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
raw=pd.read_csv('../data/punjab_wheat_district_yield_raw.csv',thousands=',',na_values=['**','-'])
rows=[];div=None
for _,r in raw.iterrows():
    n=r.district
    if n=='PUNJAB': continue
    if 'Div' in n: div=n; divval=r.iloc[2:].astype(float).values; continue
    if n=='Islamabad': continue
    for k,s in enumerate(raw.columns[2:]): rows.append((n,div,s,k,r[s],divval[k]))
L=pd.DataFrame(rows,columns=['district','division','season','t','y','div_y'])
L['y']=L.y/1000; L['div_y']=L.div_y/1000
miss=L.y.isna().sum(); L=L.dropna(subset=['y'])
flag=(L.y>5)|(np.abs(L.y/L.div_y-1)>0.6)
print('missing',miss,'flagged',flag.sum()); print(L[flag])
L['flag']=flag; L.to_csv('district_panel_long.csv',index=False)
P=L[~flag].reset_index(drop=True); print('panel',P.shape,P.district.nunique(),P.season.nunique())
# variance decomposition
def r2(cols):
    X=pd.get_dummies(P[cols].astype(str),drop_first=True).astype(float); X['c']=1; b=np.linalg.lstsq(X.values,P.y.values,rcond=None)[0]; res=P.y.values-X.values@b
    return 1-res.var()/P.y.var()
print('R2 district',round(r2(['district']),3),'season',round(r2(['season']),3),'both',round(r2(['district','season']),3))
# province-year mean series & old national data consistency
pm=P.groupby('season').y.mean(); print(pm.round(3).to_dict())
d=pd.read_csv('crop_year_dataset.csv'); w=d[d.crop=='wheat'].set_index('year').yield_ton_ha
seas={'2007-08':2008,'2008-09':2009,'2009-10':2010,'2010-11':2011,'2011-12':2012,'2012-13':2013}
pj=raw[raw.district=='PUNJAB'].iloc[0]; a=[];b=[]
for s,y in seas.items():
    if y in w.index: a.append(w[y]); b.append(float(pj[s])/1000)
print('national wheat (FAO table) vs Punjab row, harvest-year aligned:',np.round(a,2),np.round(b,2),'r=',round(np.corrcoef(a,b)[0,1],2))
# ---------- models
seasons=sorted(P.season.unique(),key=lambda s:int(s[:4])); T={s:i for i,s in enumerate(seasons)}; P['t']=P.season.map(T)
oh=OneHotEncoder(handle_unknown='ignore',sparse_output=False).fit(P[['district']])
def feats(df): return np.hstack([oh.transform(df[['district']]),df[['t']].values.astype(float)])
def fitpred(name,tr,te):
    if name=='District mean': m=tr.groupby('district').y.mean(); return te.district.map(m).fillna(tr.y.mean()).values
    if name=='District mean + common trend':
        res=tr.y-tr.district.map(tr.groupby('district').y.mean()); b=np.polyfit(tr.t,res,1); dm=te.district.map(tr.groupby('district').y.mean()).fillna(tr.y.mean())
        # recompute intercept-free: trend on residual, centred
        return (dm+np.polyval(b,te.t)).values
    if name=='Oracle season shock':
        dm=tr.groupby('district').y.mean(); base=te.district.map(dm).fillna(tr.y.mean())
        # shock = held-out season's mean deviation from district means (uses test labels: upper bound)
        shock=(te.y-base).groupby(te.season).transform('mean'); return (base+shock).values
    if name=='Ridge (district + time)': return Ridge(alpha=1.0).fit(feats(tr),tr.y).predict(feats(te))
    if name=='Random Forest (district + time)': return RandomForestRegressor(n_estimators=200,max_depth=8,random_state=42,n_jobs=1).fit(feats(tr),tr.y).predict(feats(te))
    if name=='XGBoost (district + time)': return XGBRegressor(n_estimators=200,learning_rate=0.05,max_depth=5,random_state=42,verbosity=0,n_jobs=1).fit(feats(tr),tr.y).predict(feats(te))
models=['District mean','District mean + common trend','Ridge (district + time)','Random Forest (district + time)','XGBoost (district + time)','Oracle season shock']
def splits(prot):
    if prot=='LOSO':
        for s in seasons: yield P[P.season!=s],P[P.season==s]
    elif prot=='Forward':
        for s in seasons[5:]: yield P[P.t<T[s]],P[P.season==s]
    else:
        rk=RepeatedKFold(n_splits=5,n_repeats=3,random_state=42)
        for tr,te in rk.split(P): yield P.iloc[tr],P.iloc[te]
out=[]
for prot in ['Random','LOSO','Forward']:
    for m in models:
        if prot=='Random' and m=='Oracle season shock': continue
        yy=[];pp=[];ss=[]; reps=[]
        if prot=='Random':
            sp=list(splits(prot))
            for r in range(3):
                yt=[];yp=[]
                for tr,te in sp[r*5:(r+1)*5]: yt+=list(te.y); yp+=list(fitpred(m,tr,te))
                reps.append((np.sqrt(np.mean((np.array(yt)-np.array(yp))**2)),1-np.sum((np.array(yt)-np.array(yp))**2)/np.sum((np.array(yt)-np.mean(yt))**2)))
            rm,r2v=np.mean(reps,0); lo=hi=np.nan
        else:
            for tr,te in splits(prot): yy+=list(te.y); pp+=list(fitpred(m,tr,te)); ss+=list(te.season)
            yy=np.array(yy);pp=np.array(pp);ss=np.array(ss); rm=np.sqrt(np.mean((yy-pp)**2)); r2v=1-np.sum((yy-pp)**2)/np.sum((yy-yy.mean())**2)
            rng=np.random.default_rng(0); us=np.unique(ss); bs=[]
            for _ in range(2000):
                pick=rng.choice(us,len(us)); idx=np.concatenate([np.where(ss==a)[0] for a in pick]); bs.append(np.sqrt(np.mean((yy[idx]-pp[idx])**2)))
            lo,hi=np.percentile(bs,[2.5,97.5])
        out.append((prot,m,rm,lo,hi,r2v)); print(prot,m,round(rm,3),round(lo,3) if lo==lo else '',round(hi,3) if hi==hi else '',round(r2v,3),flush=True)
pd.DataFrame(out,columns=['protocol','model','rmse','ci_lo','ci_hi','r2']).to_csv('district_results.csv',index=False)
