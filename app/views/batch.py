import pandas as pd
import streamlit as st

from core import charts
from core.features import DAY_COLS, build_features
from core.models import (MODELS, contributions, feature_columns, load_test, load_usage, predict_proba, pretty,
                         risk_band, threshold)

key = st.session_state.get('model_key', '90d')
thr = threshold(key)
cols = feature_columns(key)
n_days = MODELS[key]['n_days']

st.title('Batch scoring')
st.markdown(
    'Upload a CSV and get a churn score for every customer. Two formats are accepted:\n'
    '- **Raw daily usage**: same layout as `data/raw/main.csv` '
    '(`customer_id`, `usage_type`, `Day_1` … `Day_90`, one row per customer and service). '
    'Features are built automatically.\n'
    f'- **Features**: one row per customer with `customer_id` and the {len(cols)} feature columns '
    '(as in `data/processed/test_features.csv`).'
)


@st.cache_data
def sample_files(key):
    test = load_test(key)
    feats = test[['customer_id'] + cols].head(50).to_csv(index=False).encode()
    usage = load_usage().reset_index()
    ids = usage.customer_id.drop_duplicates().head(20)
    raw = usage[usage.customer_id.isin(ids)][['customer_id', 'usage_type'] + DAY_COLS].to_csv(index=False).encode()
    return feats, raw


with st.expander('Download example files'):
    feats_csv, raw_csv = sample_files(key)
    c1, c2 = st.columns(2)
    c1.download_button('Example features CSV (50 customers)', feats_csv, 'example_features.csv', 'text/csv')
    c2.download_button('Example raw usage CSV (20 customers)', raw_csv, 'example_raw_usage.csv', 'text/csv')

up = st.file_uploader('CSV file', type='csv')
if up is None:
    st.stop()

try:
    data = pd.read_csv(up, encoding='utf-8-sig')
except Exception as e:
    st.error(f'Could not read the file: {e}')
    st.stop()

if 'usage_type' in data.columns:
    try:
        with st.spinner('Building features from raw usage...'):
            X = build_features(data, n_days=n_days).reset_index()
    except ValueError as e:
        st.error(f'Raw usage file is not valid: {e}')
        st.stop()
    st.success(f'Built {len(cols)} features for {len(X):,} customers from {len(data):,} usage rows '
               f'(first {n_days} days).')
else:
    missing = [c for c in cols if c not in data.columns]
    if missing:
        st.error(f'{len(missing)} feature columns are missing, e.g. ' + ', '.join(missing[:5]))
        st.stop()
    X = data
    if 'customer_id' not in X.columns:
        X = X.assign(customer_id=range(len(X)))

if X[cols].isna().any().any():
    st.warning('Some feature values are missing; EBM will place them in its missing-value bin.')

out = X[['customer_id']].copy()
out['churn_proba'] = predict_proba(key, X)
out['predicted_churn'] = (out.churn_proba >= thr).astype(int)
out['risk'] = [risk_band(p, thr) for p in out.churn_proba]

contrib = contributions(key, X)
top = contrib.apply(lambda r: r.nlargest(3).index.tolist(), axis=1)
out['top_reasons'] = [', '.join(pretty(n) for n in t) for t in top]
if 'churn' in X.columns:
    out['actual_churn'] = X['churn'].values
out = out.sort_values('churn_proba', ascending=False)

c = st.columns(4)
c[0].metric('Customers scored', f'{len(out):,}')
c[1].metric('High risk', f'{(out.risk == "High").sum():,}')
c[2].metric('Medium risk', f'{(out.risk == "Medium").sum():,}')
c[3].metric('Low risk', f'{(out.risk == "Low").sum():,}')

left, right = st.columns([2, 1])
with left:
    hist = out.rename(columns={'actual_churn': 'churn'})
    st.plotly_chart(charts.probability_histogram(hist, thr), width='stretch')
with right:
    st.plotly_chart(charts.risk_donut(out), width='stretch')

show = st.multiselect('Risk bands', ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'])
view = out[out.risk.isin(show)]
st.dataframe(
    view, hide_index=True, width='stretch',
    column_config={
        'churn_proba': st.column_config.ProgressColumn('Churn probability', min_value=0, max_value=1, format='%.3f'),
        'top_reasons': st.column_config.TextColumn('Top 3 reasons towards churn', width='large'),
    },
)
st.download_button('Download results CSV', view.to_csv(index=False).encode(),
                   f'churn_scores_{key}.csv', 'text/csv', type='primary')
