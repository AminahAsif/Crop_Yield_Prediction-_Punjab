import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
exec(open('detrend.py').read().split("rows=run('LOYO')")[0])
def run2(protocol,feats,name='RF'):
    yrs=np.sort(d.year.unique()); tests=[y for y in yrs if (protocol=='LOYO' or y>=1998)]
    yy=[];yr=[];pt=[];pm=[]
    for y in tests:
        tr=d[d.year!=y] if protocol=='LOYO' else d[d.year<y]; te=d[d.year==y]
        tp,fit=trend_pred(tr,te); resid=tr.yield_ton_ha.values-fit
        yy+=list(te.yield_ton_ha); yr+=[y]*len(te); pt+=list(tp)
        pm+=list(tp+mk[name]().fit(tr[feats],resid).predict(te[feats]))
    return np.array(yy),np.array(yr),np.array(pt),np.array(pm)
rows=[]
for prot in ['LOYO','Forward']:
    for lab,feats in [('crop only',['crop_code']),('+ year (comparison)',['crop_code','year']),('+ temperature',['crop_code','avg_temp']),('+ pesticides',['crop_code','pesticides_tonnes']),('all three',F)]:
        yy,yr,pt,pm=run2(prot,feats)
        rm=np.sqrt(np.mean((yy-pm)**2)); rt=np.sqrt(np.mean((yy-pt)**2))
        rng=np.random.default_rng(1); ys=np.unique(yr); diffs=[]
        for _ in range(2000):
            pick=rng.choice(ys,len(ys)); idx=np.concatenate([np.where(yr==a)[0] for a in pick])
            diffs.append(np.sqrt(np.mean((yy[idx]-pt[idx])**2))-np.sqrt(np.mean((yy[idx]-pm[idx])**2)))
        rows.append((prot,lab,rm,rt,np.mean(diffs),np.percentile(diffs,2.5),np.percentile(diffs,97.5),np.mean(np.array(diffs)>0)))
r=pd.DataFrame(rows,columns=['protocol','features','rmse_resid_rf','rmse_trend','gain','gain_lo','gain_hi','p_gain_pos']); r.to_csv('detrend_ablation.csv',index=False); print(r.round(3).to_string())
