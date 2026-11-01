"""Churn prediction front end (EBM only).

Run from the project root:
    .venv/bin/streamlit run app/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from core.models import MODELS  # noqa: E402

st.set_page_config(page_title='Churn Predictor', page_icon='📉', layout='wide')

with st.sidebar:
    st.markdown('### Model')
    st.radio(
        'EBM model', list(MODELS), key='model_key',
        format_func=lambda k: MODELS[k]['label'],
        help='The early warning model only sees the first 30 days of usage, '
             'so it can flag churners two months earlier with lower accuracy.',
    )

views = Path(__file__).resolve().parent / 'views'
nav = st.navigation([
    st.Page(views / 'overview.py', title='Overview', icon='📊', default=True),
    st.Page(views / 'customer.py', title='Customer lookup', icon='🔎'),
    st.Page(views / 'batch.py', title='Batch scoring', icon='📂'),
    st.Page(views / 'insights.py', title='Model insights', icon='🧠'),
])

with st.sidebar:
    st.divider()
    st.caption('Explainable Boosting Machine (interpret). Red = pushes towards churn, blue = towards staying.')

nav.run()
