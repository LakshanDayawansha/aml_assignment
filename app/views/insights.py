import pandas as pd
import streamlit as st

from core import charts
from core.models import MODELS, explain_global, load_model, pretty, term_importance, train_quantiles

key = st.session_state.get('model_key', '90d')
ebm, info = load_model(key)
imp = term_importance(key)

st.title('Model insights')
st.caption(f"{MODELS[key]['label']} EBM · trained {info['trained_at']} on {info['n_train']:,} customers · "
           f"{len(info['feature_columns'])} features + {len(ebm.term_names_) - len(info['feature_columns'])} pairwise interactions")

tab1, tab2, tab3 = st.tabs(['Feature importance', 'Shape functions', 'Model comparison'])

with tab1:
    top = st.slider('Terms shown', 10, len(imp), 20)
    st.plotly_chart(charts.importance_bar(imp, top), width='stretch')
    st.caption('Mean absolute contribution to the log-odds over the training data.')

with tab2:
    st.markdown('Each EBM term is a learned curve: the model\'s prediction is the intercept plus the sum of '
                'these curves evaluated at the customer\'s feature values. This is the whole model, nothing hidden.')
    term = st.selectbox('Term', imp.index, format_func=lambda t: ('⨯ ' if ' & ' in t else '') + pretty(t))
    d = explain_global(key).data(ebm.term_names_.index(term))
    if ' & ' in term:
        st.plotly_chart(charts.interaction_heatmap(d, term), width='stretch')
    else:
        trim = st.toggle('Hide outliers (1st–99th percentile of training data)', value=True)
        lo, hi = (train_quantiles(key).loc[term, ['lo', 'hi']] if trim else (None, None))
        st.plotly_chart(charts.shape_function(d, term, lo, hi), width='stretch')
        st.caption('Above 0 = raises churn risk, below 0 = lowers it. Grey band = uncertainty across bags.')

with tab3:
    rows = []
    for k, cfg in MODELS.items():
        _, inf = load_model(k)
        rows.append({'model': cfg['label'], 'threshold': inf['decision_threshold_f1'], **inf['test_metrics']})
    st.dataframe(pd.DataFrame(rows).set_index('model').T, width='stretch')
    st.markdown(
        '- The **90 day** model sees the whole period and is close to perfect, because churners have mostly '
        'stopped using the network by the end.\n'
        '- The **30 day** model only sees the first month. It is less accurate but gives a much earlier '
        'warning, which is when a retention offer can still work.'
    )
    with st.expander('Hyperparameters'):
        st.json(info['best_params'])
