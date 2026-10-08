"""Apply frozen paired promotion gates; missing or mismatched evidence fails."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
KEYS=('joint_rmse','root_velocity_rmse','yaw_error')
WEIGHTS=np.array([.4,.3,.3])
def compare(reference,candidate,confirmation=False,criterion=None):
    criterion=criterion or json.loads((Path(__file__).resolve().parents[1]/"reports/recovery_4900/criterion_v2.json").read_text())
    for directory in (reference,candidate):
        if not (directory/'COMPLETE').is_file():raise ValueError('Incomplete evaluation: '+str(directory))
    receipts=[json.loads((p/'receipt.json').read_text()) for p in (reference,candidate)]
    for key in ('protocol_sha256','cases','code'):
        if receipts[0][key]!=receipts[1][key]:raise ValueError('Evaluation mismatch: '+key)
    rows=[json.loads((p/'results.json').read_text()) for p in (reference,candidate)]
    maps=[{(r['case'],r['mode']):r for r in rs} for rs in rows]
    if set(maps[0])!=set(maps[1]) or len(maps[0])!=len(rows[0]) or len(maps[1])!=len(rows[1]):raise ValueError('Case coverage mismatch')
    order=sorted(maps[0]);a=np.array([[maps[0][k][m] for m in KEYS] for k in order]);b=np.array([[maps[1][k][m] for m in KEYS] for k in order])
    if not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Nonfinite metrics')
    mean_ratio=b.mean(0)/a.mean(0);score=float(mean_ratio@WEIGHTS)
    ratios=b/np.maximum(a,1e-9);reasons=[]
    if score>=1.0-criterion['minimum_improvement']:reasons.append('aggregate improvement is below required positive gain')
    for i,k in enumerate(KEYS[:2]):
        if mean_ratio[i]>1+criterion['mean_joint_velocity_regression_limit']:reasons.append('mean '+k+' regressed beyond limit')
        if ratios[:,i].max()>1+criterion['per_case_joint_velocity_regression_limit']:reasons.append('one case '+k+' regressed beyond limit')
    falls=[list(k) for k in order if maps[1][k]['fall_proxy'] and not maps[0][k]['fall_proxy']]
    if falls:reasons.append('new fall proxy')
    modes={}
    for mode in ('raw','wm'):
        indices=[i for i,k in enumerate(order) if k[1]==mode]
        modes[mode]=dict(score=float((b[indices].mean(0)/a[indices].mean(0))@WEIGHTS),mean_ratio=dict(zip(KEYS,(b[indices].mean(0)/a[indices].mean(0)).tolist())))
    motions=sorted({k[0] for k in order});rng=np.random.default_rng(4900);bootstrap=[]
    groups=[[i for i,k in enumerate(order) if k[0]==motion] for motion in motions]
    for _ in range(10000):
        sampled=np.concatenate([groups[i] for i in rng.integers(len(groups),size=len(groups))])
        bootstrap.append(float((b[sampled].mean(0)/a[sampled].mean(0))@WEIGHTS))
    upper=float(np.quantile(bootstrap,.95))
    if confirmation and upper>=criterion['confirmation_bootstrap_upper_ratio']:reasons.append('confirmation paired motion bootstrap upper bound >=1')
    return dict(criterion=criterion,accepted=not reasons,reasons=reasons,score_ratio=score,improvement_percent=(1-score)*100,
        mean_ratio=dict(zip(KEYS,mean_ratio.tolist())),worst_ratio=dict(zip(KEYS,ratios.max(0).tolist())),new_falls=falls,
        baseline_mean=dict(zip(KEYS,a.mean(0).tolist())),candidate_mean=dict(zip(KEYS,b.mean(0).tolist())),modes=modes,
        paired_motion_bootstrap_upper95=upper,case_count=len(order),per_case=[dict(case=k[0],mode=k[1],ratios=dict(zip(KEYS,ratios[i].tolist()))) for i,k in enumerate(order)])
def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True);p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--confirmation',action='store_true');p.add_argument('--criterion',type=Path,default=Path(__file__).resolve().parents[1]/'reports/recovery_4900/criterion_v2.json');a=p.parse_args()
    result=compare(a.reference,a.candidate,a.confirmation,json.loads(a.criterion.read_text()));a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='per_case'},indent=2))
if __name__=='__main__':main()
