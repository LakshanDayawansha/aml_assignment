import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from core.features import DAY_COLS, SERVICES

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / 'models'
DATA_DIR = ROOT / 'data'
CACHE_DIR = ROOT / 'app' / 'cache'

MODELS = {
    '90d': {
        'label': 'Full period (90 days)',
        'file': 'ebm_churn',
        'test': 'test_features.csv',
        'n_days': 90,
    },
    '30d': {
        'label': 'Early warning (first 30 days)',
        'file': 'ebm_churn_30d',
        'test': 'test_features_30d.csv',
        'n_days': 30,
    },
}

SERVICE_LABELS = {
    'app_facebook': 'Facebook', 'app_helakuru': 'Helakuru', 'app_other': 'Other apps',
    'app_tiktok': 'TikTok', 'app_whatsapp': 'WhatsApp', 'app_youtube': 'YouTube',
    'pack_data': 'Data pack', 'pack_vas': 'VAS pack',
    'voice_d2d_incoming': 'Voice in (same net)', 'voice_d2d_outgoing': 'Voice out (same net)',
    'voice_d2nd_outgoing': 'Voice out (other net)', 'voice_nd2d_incoming': 'Voice in (other net)',
    'all': 'All services',
}

FEATURE_LABELS = {
    'ratio_last30_first30': 'trend (last / first third)',
    'ratio_last30_first30_min': 'trend, min',
    'ratio_last30_first30_mean': 'trend, mean',
    'ratio_last30_first30_max': 'trend, max',
    'active_days': 'active days',
    'active_days_min': 'active days, min',
    'active_days_mean': 'active days, mean',
    'active_days_max': 'active days, max',
    'days_since_active': 'days since active',
    'days_since_active_min': 'days since active, min',
    'days_since_active_mean': 'days since active, mean',
    'days_since_active_max': 'days since active, max',
    'total': 'total usage',
    'last30_share': 'share of usage in last third',
}

OTHER_LABELS = {
    'data_voice_ratio': 'Data / voice ratio',
    'n_services_active_last7': 'Services used in last 7 days',
}


def pretty(name):
    """'app_youtube__active_days' -> 'YouTube: active days'. Works for interaction terms too."""
    if ' & ' in name:
        return ' × '.join(pretty(n) for n in name.split(' & '))
    if name in OTHER_LABELS:
        return OTHER_LABELS[name]
    if '__' in name:
        svc, feat = name.split('__', 1)
        return SERVICE_LABELS.get(svc, svc) + ': ' + FEATURE_LABELS.get(feat, feat)
    return name


@st.cache_resource
def load_model(key):
    cfg = MODELS[key]
    ebm = joblib.load(MODEL_DIR / (cfg['file'] + '.joblib'))
    with open(MODEL_DIR / (cfg['file'] + '_info.json')) as f:
        info = json.load(f)
    # the 30 day model was evaluated with predict(), i.e. a 0.5 cut-off
    info.setdefault('decision_threshold_f1', 0.5)
    return ebm, info


def threshold(key):
    return load_model(key)[1]['decision_threshold_f1']


def feature_columns(key):
    return load_model(key)[1]['feature_columns']


def predict_proba(key, X):
    ebm, info = load_model(key)
    return ebm.predict_proba(X[info['feature_columns']])[:, 1]


def contributions(key, X):
    """Log-odds contribution of every term (features and pairs) for each row of X."""
    ebm, info = load_model(key)
    scores = ebm.eval_terms(X[info['feature_columns']])
    return pd.DataFrame(scores, columns=ebm.term_names_, index=X.index)


def intercept(key):
    return float(load_model(key)[0].intercept_[0])


def risk_band(p, thr):
    if p >= thr:
        return 'High'
    if p >= 0.2:
        return 'Medium'
    return 'Low'


@st.cache_data
def load_test(key):
    """Test set with model scores. These customers were not used for training."""
    df = pd.read_csv(DATA_DIR / 'processed' / MODELS[key]['test'])
    df['churn_proba'] = predict_proba(key, df)
    thr = threshold(key)
    df['predicted'] = (df.churn_proba >= thr).astype(int)
    df['risk'] = [risk_band(p, thr) for p in df.churn_proba]
    return df


@st.cache_data
def term_importance(key):
    ebm = load_model(key)[0]
    return pd.Series(ebm.term_importances(), index=ebm.term_names_).sort_values(ascending=False)


@st.cache_resource
def explain_global(key):
    return load_model(key)[0].explain_global()


@st.cache_data
def train_quantiles(key, q=(0.01, 0.99)):
    """Feature ranges in the training data, used for slider limits and plot ranges."""
    name = 'train_features_30d.csv' if key == '30d' else 'train_features.csv'
    df = pd.read_csv(DATA_DIR / 'processed' / name, usecols=feature_columns(key))
    return df.quantile(list(q)).T.set_axis(['lo', 'hi'], axis=1).join(df.agg(['min', 'max', 'median']).T)


@st.cache_resource(show_spinner='Preparing raw usage for test customers (first run only, ~30 s)...')
def load_usage():
    """Raw daily usage for test customers, cached as parquet so the 400 MB csv is read only once.

    Saved as a few smaller part files so each one is easy to push to GitHub."""
    path = CACHE_DIR / 'test_usage'
    if not any(path.glob('*.parquet')):
        path.mkdir(parents=True, exist_ok=True)
        ids = set(pd.read_csv(DATA_DIR / 'raw' / 'test' / 'test_cxid.csv', encoding='utf-8-sig').customer_id)
        parts = []
        dtypes = {c: np.float32 for c in DAY_COLS}
        for chunk in pd.read_csv(DATA_DIR / 'raw' / 'main.csv', dtype=dtypes, chunksize=100_000):
            parts.append(chunk[chunk.customer_id.isin(ids)])
        usage = pd.concat(parts).sort_values('customer_id')
        for i, rows in enumerate(np.array_split(np.arange(len(usage)), 4)):
            usage.iloc[rows].to_parquet(path / f'part-{i}.parquet', index=False, compression='zstd', compression_level=19)
    usage = pd.concat([pd.read_parquet(p) for p in sorted(path.glob('*.parquet'))])
    return usage.set_index('customer_id').sort_index()


def customer_usage(customer_id):
    usage = load_usage()
    if customer_id not in usage.index:
        return None
    rows = usage.loc[[customer_id]].set_index('usage_type').reindex(SERVICES).fillna(0)
    return rows[DAY_COLS]
