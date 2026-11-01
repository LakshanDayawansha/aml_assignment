import pandas as pd
import streamlit as st

from core import charts
from core.models import MODELS, contributions, customer_usage, intercept, load_test, pretty, threshold

key = st.session_state.get('model_key', '90d')
df = load_test(key)
thr = threshold(key)
n_days = MODELS[key]['n_days']

st.title('Customer lookup')

counts = df.risk.value_counts()
f1, f2, f3 = st.columns([1, 1, 3])
band = f1.selectbox('Filter by risk', ['All', 'High', 'Medium', 'Low'],
                    format_func=lambda b: f'{b} ({counts.get(b, 0) if b != "All" else len(df):,})')
order = f2.selectbox('Order', ['Mixed', 'Highest risk first', 'Lowest risk first'])
pool = df if band == 'All' else df[df.risk == band]
if order == 'Mixed':
    pool = pool.sample(frac=1, random_state=0)
else:
    pool = pool.sort_values('churn_proba', ascending=(order == 'Lowest risk first'))
if pool.empty:
    st.info('No customers in this band.')
    st.stop()
proba = dict(zip(pool.customer_id, pool.churn_proba))
cid = f3.selectbox(
    f'Customer ({len(pool):,}, type to search)', pool.customer_id,
    format_func=lambda c: f'{c[:16]}…  ({proba[c]:.1%})',
)

row = df[df.customer_id == cid]
p = float(row.churn_proba.iloc[0])
actual = int(row.churn.iloc[0])

g, s = st.columns([1, 2])
with g:
    st.plotly_chart(charts.gauge(p, thr), width='stretch')
with s:
    st.markdown(f'**Customer** `{cid}`')
    c = st.columns(3)
    c[0].metric('Risk band', row.risk.iloc[0])
    c[1].metric('Prediction', 'Churn' if p >= thr else 'Stay')
    c[2].metric('Actual outcome', 'Churned' if actual else 'Stayed',
                'correct' if (p >= thr) == bool(actual) else 'missed', delta_color='normal' if (p >= thr) == bool(actual) else 'inverse')
    c = st.columns(3)
    c[0].metric('Mean trend (all services)', f"{row['all__ratio_last30_first30_mean'].iloc[0]:.2f}")
    c[1].metric('Days since any activity', int(row['all__days_since_active_min'].iloc[0]))
    c[2].metric('Services used, last 7 days', f"{int(row['n_services_active_last7'].iloc[0])} / 12")

st.subheader('Why this prediction?')
contrib = contributions(key, row).iloc[0]
top_n = st.slider('Terms shown', 5, 30, 12)
st.plotly_chart(charts.contribution_bar(contrib, row.iloc[0].to_dict(), top=top_n), width='stretch')
total = intercept(key) + contrib.sum()
st.caption(
    f'EBM is additive: intercept ({intercept(key):.2f}) + sum of all {len(contrib)} term contributions '
    f'({contrib.sum():.2f}) = {total:.2f} log-odds → probability {p:.3f}.'
)

drivers = contrib.sort_values()
c1, c2 = st.columns(2)
with c1:
    st.markdown('**Top reasons towards churn**')
    for n, v in drivers[::-1].head(5).items():
        if v > 0:
            st.markdown(f'- {pretty(n)} (+{v:.2f})')
with c2:
    st.markdown('**Top reasons towards staying**')
    for n, v in drivers.head(5).items():
        if v < 0:
            st.markdown(f'- {pretty(n)} ({v:.2f})')

st.subheader('Usage over 90 days')
usage = customer_usage(cid)
if usage is None:
    st.info('No raw usage found for this customer.')
else:
    tab1, tab2 = st.tabs(['Daily usage', 'Activity map'])
    with tab1:
        group = st.radio('Services', ['All', 'Apps', 'Packs', 'Voice'], horizontal=True)
        prefix = {'All': None, 'Apps': 'app_', 'Packs': 'pack_', 'Voice': 'voice_'}[group]
        st.plotly_chart(charts.usage_timeline(usage, n_days, prefix), width='stretch')
    with tab2:
        st.plotly_chart(charts.activity_heatmap(usage), width='stretch')
        st.caption('Blue = the service was used that day. Churners tend to switch services off one by one.')

with st.expander('All feature values'):
    vals = row.drop(columns=['customer_id', 'churn', 'churn_proba', 'predicted', 'risk']).iloc[0]
    table = pd.DataFrame({'feature': [pretty(n) for n in vals.index], 'column': vals.index,
                          'value': vals.values, 'contribution': contrib.reindex(vals.index).values})
    st.dataframe(table, hide_index=True, width='stretch',
                 column_config={'contribution': st.column_config.NumberColumn(format='%.3f')})
