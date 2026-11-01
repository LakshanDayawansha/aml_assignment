import pandas as pd
import streamlit as st

from core import charts
from core.models import MODELS, load_test, threshold

key = st.session_state.get('model_key', '90d')
df = load_test(key)
thr = threshold(key)

st.title('Customer churn overview')
st.caption(f"{MODELS[key]['label']} EBM, scored on the {len(df):,} held-out test customers.")

c = st.columns(3)
c[0].metric('Customers', f'{len(df):,}')
c[1].metric('Predicted to churn', f'{df.predicted.sum():,}', f'{df.predicted.mean():.1%} of base', delta_color='off')
c[2].metric('Actual churners', f'{df.churn.sum():,}', f'{df.churn.mean():.1%} of base', delta_color='off')

left, right = st.columns([2, 1])
with left:
    st.subheader('Predicted probability by actual outcome')
    st.plotly_chart(charts.probability_histogram(df, thr), width='stretch')
with right:
    st.subheader('Risk bands')
    st.plotly_chart(charts.risk_donut(df), width='stretch')
    st.caption(f'High ≥ {thr:.2f} (decision threshold) · Medium 0.20–{thr:.2f} · Low < 0.20')

st.subheader('Customers')
counts = df.risk.value_counts()
f1, f2, f3 = st.columns(3)
bands = f1.multiselect('Risk bands', ['High', 'Medium', 'Low'], default=['High', 'Medium', 'Low'],
                       format_func=lambda b: f'{b} ({counts.get(b, 0):,})')
order = f2.selectbox('Order', ['Mixed', 'Highest risk first', 'Lowest risk first'])
n = f3.slider('Rows', 10, 200, 25, step=5)
pool = df[df.risk.isin(bands)]
if order == 'Mixed':
    # same number of rows from each selected band, so low risk customers are visible too
    per_band = -(-n // max(len(bands), 1))
    parts = [pool[pool.risk == b] for b in bands]
    pool = pd.concat([p.sample(min(len(p), per_band), random_state=0) for p in parts]).sample(frac=1, random_state=0)
else:
    pool = pool.sort_values('churn_proba', ascending=(order == 'Lowest risk first'))
top = pool.head(n)[['customer_id', 'churn_proba', 'risk', 'churn', 'all__ratio_last30_first30_mean',
                    'all__days_since_active_min', 'n_services_active_last7']]
st.dataframe(
    top, hide_index=True, width='stretch',
    column_config={
        'customer_id': st.column_config.TextColumn('Customer'),
        'churn_proba': st.column_config.ProgressColumn('Churn probability', min_value=0, max_value=1, format='%.3f'),
        'risk': 'Risk',
        'churn': st.column_config.CheckboxColumn('Actually churned'),
        'all__ratio_last30_first30_mean': st.column_config.NumberColumn('Mean trend', format='%.2f'),
        'all__days_since_active_min': 'Days since any activity',
        'n_services_active_last7': 'Services used last 7 days',
    },
)
st.caption('Open **Customer lookup** to see why a customer is flagged.')
