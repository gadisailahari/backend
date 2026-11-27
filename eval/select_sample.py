import argparse, sys
from pathlib import Path
import numpy as np, pandas as pd
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'app/rag'))
from app.ml.risk_model import run_risk_prediction
from safety import egfr_ckdepi_2021
FEATURES = ['RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'BMXBMI', 'BMXWAIST', 'DR1TKCAL', 'DR1TSODI', 'DR1TPOTA', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT', 'DR1TALCO', 'LBXSCR', 'SMQ020', 'PAQ650', 'PAQ665', 'ALQ121']

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bmx', required=True, help='BMX_J.XPT from CDC (height and weight)')
    ap.add_argument('--demo', required=True, help='DEMO_J.XPT from CDC (pregnancy status)')
    ap.add_argument('--table', default=str(REPO / 'training/data/processed_2017_full.csv'))
    ap.add_argument('--seed', type=int, default=20261126)
    ap.add_argument('--per_cell', type=int, default=20)
    ap.add_argument('--out', default=str(REPO / 'app/eval_data/sample_120.csv'))
    ap.add_argument('--audit', default=str(REPO / 'app/eval_data/sample_selection_audit.txt'))
    a = ap.parse_args()
    d = pd.read_csv(a.table)
    d['SEQN'] = d['SEQN'].astype(int)
    n0 = len(d)
    log = [f'Processed NHANES 2017-18 table: {n0} adults']
    d = d[(d.RIDAGEYR >= 18) & (d.RIDAGEYR <= 80)]
    log.append(f'Age 18-80: {len(d)}')
    d = d.dropna(subset=FEATURES)
    log.append(f'Complete data for all 18 model inputs: {len(d)}')
    bmx = pd.read_sas(a.bmx, format='xport')[['SEQN', 'BMXHT', 'BMXWT']]
    bmx['SEQN'] = bmx['SEQN'].astype(int)
    d = d.merge(bmx, on='SEQN', how='left').dropna(subset=['BMXHT', 'BMXWT'])
    log.append(f'With measured height and weight: {len(d)}')
    demo = pd.read_sas(a.demo, format='xport')[['SEQN', 'RIDEXPRG']]
    demo['SEQN'] = demo['SEQN'].astype(int)
    d = d.merge(demo, on='SEQN', how='left')
    d = d[d.RIDEXPRG != 1]
    log.append(f'Not pregnant (the tool gives no plan to pregnant users): {len(d)}')
    d['egfr'] = [egfr_ckdepi_2021(r.LBXSCR, r.RIDAGEYR, int(r.RIAGENDR)) for r in d.itertuples()]
    d = d[d.egfr >= 60]
    log.append(f'eGFR 60 or higher (the tool refers lower values to a clinician with no plan): {len(d)}')
    d['risk_pct'] = [round(run_risk_prediction({k: float(getattr(r, k)) for k in FEATURES})['risk_probability'] * 100, 1) for r in d.itertuples()]
    d['stratum'] = np.where(d.risk_pct >= 50, 'higher', np.where(d.risk_pct >= 33, 'moderate', 'lower'))
    d['sex_group'] = np.where(d.RIAGENDR == 1, 'male', 'female')
    log.append('Available per cell (stratum, sex):')
    cnt = d.groupby(['stratum', 'sex_group']).size()
    for (s, g), c in cnt.items():
        log.append(f'  {s:9s} {g:7s} {c}')
    parts = []
    for s in ('lower', 'moderate', 'higher'):
        for g in ('female', 'male'):
            cell = d[(d.stratum == s) & (d.sex_group == g)]
            if len(cell) < a.per_cell:
                sys.exit('\n'.join(log) + f'\nNot enough participants in {s}/{g}: {len(cell)} < {a.per_cell}')
            parts.append(cell.sample(n=a.per_cell, random_state=a.seed))
    out = pd.concat(parts)
    out['SEQN'] = out['SEQN'].astype(int)
    for k in ('RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'SMQ020', 'PAQ650', 'PAQ665'):
        out[k] = out[k].astype(int)
    cols = ['SEQN', 'stratum', 'sex_group'] + FEATURES + ['BMXHT', 'BMXWT', 'egfr', 'risk_pct', 'mean_sbp', 'mean_dbp', 'hypertension']
    out[cols].round(3).to_csv(a.out, index=False)
    log.append(f'Seed {a.seed}. Selected {len(out)} participants ({a.per_cell} per stratum and sex). Written to {a.out}')
    Path(a.audit).write_text('\n'.join(log) + '\n')
    print('\n'.join(log))
if __name__ == '__main__':
    main()
