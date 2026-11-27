import csv, hashlib, io, json, os, threading, time, traceback
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from app.database import get_conn
REPO = Path(__file__).resolve().parent.parent
EVAL_DIR = Path(__file__).resolve().parent / 'eval_data'
SAMPLE = EVAL_DIR / 'sample_120.csv'
DELAY_S = float(os.environ.get('EVAL_DELAY_S', '6'))
RUNS = (1, 2)
router = APIRouter(prefix='/eval')
_state = {'state': 'idle', 'done': 0, 'total': 0, 'failed': 0, 'error': None, 'started': None}
_lock = threading.Lock()
INPUTS = ['RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'INDFMPIR', 'BMXBMI', 'BMXWAIST', 'DR1TKCAL', 'DR1TSODI', 'DR1TPOTA', 'DR1TPROT', 'DR1TCARB', 'DR1TTFAT', 'DR1TALCO', 'LBXSCR', 'SMQ020', 'PAQ650', 'PAQ665', 'ALQ121']

def _admin(key: Optional[str]=Query(default=None)):
    from app.plan_routes import ADMIN_KEY
    if ADMIN_KEY and key != ADMIN_KEY:
        raise HTTPException(401, 'Missing or invalid admin key')
    return True

def init_tables():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute('CREATE TABLE IF NOT EXISTS eval_plans (\n                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, seqn INTEGER NOT NULL, run INTEGER NOT NULL,\n                config_hash TEXT, row_json TEXT NOT NULL, UNIQUE (seqn, run))')
            cur.execute('CREATE TABLE IF NOT EXISTS eval_freeze (id INTEGER PRIMARY KEY, frozen_json TEXT NOT NULL)')
        c.commit()

def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def current_config():
    import llm, pipeline
    files = sorted(list((REPO / 'app/rag').glob('*.py')) + list((REPO / 'app/ml').glob('*.py')) + [REPO / 'app/evaluation.py'])
    data = [REPO / 'app/rag/data/menu.json', REPO / 'app/rag/data/key_facts.json', REPO / 'app/rag/data/corpus.jsonl', REPO / 'app/rag/data/fndds_2017_18_per100g.csv', REPO / 'app/rag/data/fndds_2017_18_portions.csv', REPO / 'app/ml/xgb_hypertension_model_2017.json']
    return {'model': llm.MODEL, 'provider': 'Groq chat-completions API (https://api.groq.com/openai/v1/chat/completions), called from code', 'temperature': 0.2, 'reasoning_effort': 'low' if llm.MODEL.startswith('openai/gpt-oss') else None, 'max_tokens': 2000, 'json_mode': True, 'max_attempts_per_plan': 3, 'rule_based_fallback': False, 'prompt_version': pipeline.PROMPT_VERSION, 'system_prompt_sha256': hashlib.sha256(pipeline.SYSTEM.encode()).hexdigest(), 'code_sha256': {str(f.relative_to(REPO)): _sha(f) for f in files if f.name != '__init__.py' or True}, 'data_sha256': {str(f.relative_to(REPO)): _sha(f) for f in data if f.exists()}, 'sample_sha256': _sha(SAMPLE) if SAMPLE.exists() else None}

def _config_hash(cfg):
    cfg = {k: v for k, v in cfg.items() if k != 'frozen_at'}
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]

def frozen_config():
    init_tables()
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute('SELECT frozen_json FROM eval_freeze WHERE id = 1')
            r = cur.fetchone()
    return json.loads(r['frozen_json']) if r else None

def sample_rows():
    if not SAMPLE.exists():
        raise HTTPException(409, 'app/eval_data/sample_120.csv is missing. Run eval/select_sample.py first and deploy the file.')
    with open(SAMPLE, newline='') as f:
        return list(csv.DictReader(f))

def _f(x):
    try:
        return float(x) if x not in (None, '') else None
    except ValueError:
        return None

def _screen_text(rk, ex):
    lines = [f"Risk estimate: {round(rk['risk_probability'] * 100)}% ({rk['risk_label']})", 'Main factors:']
    lines += [f"- {f}: {('raises' if s > 0 else 'lowers')} the estimate" for f, s in ex['top_factors'][:3]]
    return '\n'.join(lines)

def run_one(r, run_no, cfg_hash):
    from app.plan_routes import get_retriever, _llm
    import pipeline
    from safety import egfr_ckdepi_2021
    from app.ml.risk_model import run_risk_prediction, run_explainability
    import llm as llm_mod
    profile = {k: _f(r[k]) for k in INPUTS}
    for k in ('RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'SMQ020', 'PAQ650', 'PAQ665'):
        profile[k] = int(profile[k])
    height, weight = (_f(r['BMXHT']), _f(r['BMXWT']))
    prefs = {'cuisine': None, 'constraint': None, 'dislikes': [], 'conditions': [], 'activities': [], 'height_cm': height, 'weight_kg': weight}
    rk = run_risk_prediction(profile)
    ex = run_explainability(profile)
    st = pipeline.build_structured_input(profile, rk['risk_probability'], ex['top_factors'], prefs)
    prefs['fitness'] = st['fitness']
    out = {'SEQN': int(r['SEQN']), 'run': run_no, 'stratum': r.get('stratum'), 'sex_group': r.get('sex_group'), **{k: profile[k] for k in INPUTS}, 'height_cm': height, 'weight_kg': weight, 'egfr': round(egfr_ckdepi_2021(profile['LBXSCR'], profile['RIDAGEYR'], profile['RIAGENDR']), 1), 'cuisine_preference': 'none given (NHANES has no cuisine data)', 'diet_pattern': 'no restriction', 'disliked_foods': 'none', 'activity_level': (st['ENERGY_TARGET'] or {}).get('activity_level'), 'risk_pct': round(rk['risk_probability'] * 100, 1), 'risk_category': st['risk_tier'], 'risk_screen_text': _screen_text(rk, ex), 'measured_mean_sbp': _f(r.get('mean_sbp')), 'measured_mean_dbp': _f(r.get('mean_dbp')), 'hypertension_label': r.get('hypertension'), 'config_hash': cfg_hash, 'model': llm_mod.MODEL, 'run_started': time.strftime('%Y-%m-%d %H:%M:%S')}
    et = st['ENERGY_TARGET'] or {}
    pt = st.get('PROTEIN_TARGET_G') or {}
    out.update({'target_energy_kcal': et.get('energy_kcal'), 'target_bmr_kcal': et.get('bmr_kcal'), 'target_activity_factor': et.get('activity_factor'), 'target_deficit_kcal': et.get('deficit_kcal'), 'target_carb_g': et.get('carb_g'), 'target_protein_g': et.get('protein_g'), 'target_protein_range_g': f"{pt.get('low_g')}-{pt.get('high_g')}" if pt else None, 'target_fat_g': et.get('fat_g'), 'target_satfat_max_g': et.get('satfat_max_g'), 'target_fibre_min_g': et.get('fibre_min_g'), 'target_sodium_max_mg': et.get('sodium_max_mg'), 'reference_potassium_mg': et.get('potassium_reference_mg'), 'needs_clinician': st['needs_clinician'], 'creatinine_unknown': st['creatinine_unknown']})

    def paced(m, **kw):
        time.sleep(DELAY_S)
        return _llm(m, **kw)
    t0 = time.time()
    service_errors = []
    for tryno in range(1, 4):
        try:
            res = pipeline.generate_plan(st, get_retriever(), paced, allow_rules=False)
            break
        except Exception as e:
            service_errors.append(f"service try {tryno}: {str(getattr(e, 'detail', e))[:200]}")
            res = {'status': 'error', 'attempts': 0, 'plan': None, 'history': [{'attempt': 0, 'violations': service_errors[:]}], 'trace': None}
            time.sleep(0 if DELAY_S == 0 else 20)
    out['status'] = res['status']
    out['attempts'] = res.get('attempts')
    out['seconds'] = round(time.time() - t0, 1)
    out['built_by'] = res.get('built_by') or ('model' if res['status'] == 'approved_by_checker' else '')
    out['errors'] = ' | '.join((f"attempt {h['attempt']}: " + '; '.join(h['violations']) for h in res.get('history', [])))
    meta = getattr(getattr(llm_mod, 'LAST', None), 'meta', None)
    out['llm_response_meta'] = json.dumps(meta) if meta else ''
    if res['status'] == 'approved_by_checker':
        v = res['view']
        T = res['totals']
        out['plan_text'] = pipeline.render_plan_text(v)
        out['plan_json'] = json.dumps(res['plan'])
        out.update({'total_energy_kcal': T['kcal'], 'total_protein_g': T['protein_g'], 'total_carb_g': T['carb_g'], 'total_fat_g': T['fat_g'], 'total_satfat_g': T['satfat_g'], 'total_fibre_g': T['fiber_g'], 'total_sodium_mg_worst_case': T['sodium_mg_worst_case'], 'total_sodium_mg_average': T['sodium_mg_average'], 'total_potassium_mg': T['potassium_mg']})
    else:
        out['plan_text'] = res.get('message') or ''
        out['plan_json'] = ''
    tr = res.get('trace') if isinstance(res.get('trace'), dict) else {}
    out['user_prompt'] = tr.get('user_prompt', '')
    out['raw_model_replies'] = json.dumps(tr.get('replies'))[:60000]
    return out

def _done():
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute('SELECT seqn, run FROM eval_plans')
            return {(r['seqn'], r['run']) for r in cur.fetchall()}

def _worker():
    try:
        init_tables()
        cfg = current_config()
        fz = frozen_config()
        if not fz or _config_hash(cfg) != _config_hash(fz):
            _state.update(state='error', error='The code, prompt, model or data differ from the frozen version. Nothing was run.')
            return
        h = _config_hash(cfg)
        rows = sample_rows()
        done = _done()
        todo = [(r, n) for n in RUNS for r in rows if (int(r['SEQN']), n) not in done]
        _state.update(state='running', total=len(rows) * len(RUNS), done=len(done), failed=0, error=None, started=time.strftime('%Y-%m-%d %H:%M:%S'))
        for r, n in todo:
            try:
                o = run_one(r, n, h)
            except Exception as e:
                o = {'SEQN': int(r['SEQN']), 'run': n, 'status': 'error', 'errors': f'{type(e).__name__}: {e}', 'config_hash': h, 'plan_text': '', 'plan_json': ''}
                print('EVAL ERROR', traceback.format_exc(), flush=True)
            if o.get('status') != 'approved_by_checker':
                _state['failed'] += 1
            with get_conn() as c:
                with c.cursor() as cur:
                    cur.execute('INSERT INTO eval_plans (seqn, run, config_hash, row_json) VALUES (%s,%s,%s,%s) ON CONFLICT (seqn, run) DO NOTHING', (o['SEQN'], o['run'], h, json.dumps(o)))
                c.commit()
            _state['done'] += 1
        _state['state'] = 'done'
    except Exception as e:
        _state.update(state='error', error=f'{type(e).__name__}: {e}')
        print('EVAL WORKER ERROR', traceback.format_exc(), flush=True)

@router.get('/config')
def config(_a: bool=Depends(_admin)):
    cfg = current_config()
    return {'config_hash': _config_hash(cfg), 'frozen': bool(frozen_config()), 'matches_freeze': bool(frozen_config()) and _config_hash(frozen_config()) == _config_hash(cfg), 'config': cfg}

@router.post('/freeze')
def freeze(_a: bool=Depends(_admin)):
    if frozen_config():
        raise HTTPException(409, 'Already frozen. It cannot be changed once the evaluation has started.')
    cfg = current_config()
    cfg['frozen_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute('INSERT INTO eval_freeze (id, frozen_json) VALUES (1, %s)', (json.dumps(cfg, sort_keys=True),))
        c.commit()
    return {'frozen': True, 'config_hash': _config_hash(cfg)}

@router.get('/frozen')
def frozen(_a: bool=Depends(_admin)):
    f = frozen_config()
    if not f:
        raise HTTPException(404, 'Not frozen yet')
    return f

@router.post('/start')
def start(_a: bool=Depends(_admin)):
    with _lock:
        if _state['state'] == 'running':
            raise HTTPException(409, 'Already running')
        _state.update(state='starting', error=None)
    threading.Thread(target=_worker, daemon=True).start()
    return {'started': True}

@router.get('/status')
def status(_a: bool=Depends(_admin)):
    init_tables()
    return {**_state, 'rows_saved': len(_done())}
COLUMNS = ['SEQN', 'run', 'stratum', 'sex_group'] + INPUTS + ['height_cm', 'weight_kg', 'measured_mean_sbp', 'measured_mean_dbp', 'hypertension_label', 'egfr', 'cuisine_preference', 'diet_pattern', 'disliked_foods', 'activity_level', 'risk_pct', 'risk_category', 'target_energy_kcal', 'target_bmr_kcal', 'target_activity_factor', 'target_deficit_kcal', 'target_carb_g', 'target_protein_g', 'target_protein_range_g', 'target_fat_g', 'target_satfat_max_g', 'target_fibre_min_g', 'target_sodium_max_mg', 'reference_potassium_mg', 'needs_clinician', 'creatinine_unknown', 'status', 'attempts', 'built_by', 'errors', 'total_energy_kcal', 'total_protein_g', 'total_carb_g', 'total_fat_g', 'total_satfat_g', 'total_fibre_g', 'total_sodium_mg_worst_case', 'total_sodium_mg_average', 'total_potassium_mg', 'risk_screen_text', 'plan_text', 'plan_json', 'user_prompt', 'raw_model_replies', 'model', 'llm_response_meta', 'config_hash', 'run_started', 'seconds']

@router.get('/export.csv')
def export_csv(_a: bool=Depends(_admin)):
    init_tables()
    with get_conn() as c:
        with c.cursor() as cur:
            cur.execute('SELECT row_json FROM eval_plans ORDER BY seqn, run')
            rows = [json.loads(r['row_json']) for r in cur.fetchall()]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, extrasaction='ignore')
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return Response(buf.getvalue().encode('utf-8-sig'), media_type='text/csv', headers={'Content-Disposition': 'attachment; filename=pulse_and_plate_evaluation.csv'})
