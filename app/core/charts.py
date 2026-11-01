import numpy as np
import pandas as pd
import plotly.graph_objects as go

from core.features import DAY_COLS, short_name
from core.models import SERVICE_LABELS, pretty

UP = '#d1495b'     # pushes towards churn
DOWN = '#2e86ab'   # pushes towards staying
GREY = '#8d99ae'
RISK_COLORS = {'High': UP, 'Medium': '#edae49', 'Low': '#66a182'}


def _layout(fig, height=400, **kw):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), **kw)
    return fig


def gauge(p, thr):
    fig = go.Figure(go.Indicator(
        mode='gauge+number',
        value=p * 100,
        number={'suffix': '%', 'valueformat': '.1f'},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': UP if p >= thr else DOWN},
            'steps': [
                {'range': [0, 20], 'color': 'rgba(102,161,130,0.25)'},
                {'range': [20, thr * 100], 'color': 'rgba(237,174,73,0.25)'},
                {'range': [thr * 100, 100], 'color': 'rgba(209,73,91,0.25)'},
            ],
            'threshold': {'line': {'color': 'black', 'width': 3}, 'value': thr * 100},
        },
        title={'text': 'Churn probability'},
    ))
    return _layout(fig, height=260)


def contribution_bar(contrib, values=None, top=12):
    """Horizontal bar of the largest log-odds contributions for one customer."""
    s = contrib[contrib.abs().sort_values(ascending=False).index[:top]][::-1]
    labels = [pretty(n) for n in s.index]
    if values is not None:
        labels = [f'{l} = {_fmt(values.get(n))}' if n in values else l for l, n in zip(labels, s.index)]
    fig = go.Figure(go.Bar(
        x=s.values, y=labels, orientation='h',
        marker_color=[UP if v > 0 else DOWN for v in s.values],
        hovertemplate='%{y}<br>contribution: %{x:.3f}<extra></extra>',
    ))
    fig.add_vline(x=0, line_color='black', line_width=1)
    return _layout(fig, height=max(320, 28 * len(s) + 60),
                   xaxis_title='log-odds contribution (red = towards churn, blue = towards staying)')


def _fmt(v):
    if v is None:
        return ''
    v = float(v)
    return f'{v:.0f}' if v == int(v) and abs(v) < 1e6 else f'{v:.3g}'


def usage_timeline(usage, n_days=90, group=None):
    """Daily usage lines for one customer. usage: services x Day_1..Day_90."""
    fig = go.Figure()
    days = np.arange(1, 91)
    for svc, row in usage.iterrows():
        name = short_name(svc)
        if group and not name.startswith(group):
            continue
        fig.add_trace(go.Scatter(x=days, y=row[DAY_COLS].values, mode='lines',
                                 name=SERVICE_LABELS.get(name, name), line=dict(width=1.5)))
    k = n_days // 3
    fig.add_vrect(x0=0.5, x1=k + 0.5, fillcolor=DOWN, opacity=0.08, line_width=0,
                  annotation_text='first third', annotation_position='top left')
    fig.add_vrect(x0=n_days - k + 0.5, x1=n_days + 0.5, fillcolor=UP, opacity=0.08, line_width=0,
                  annotation_text='last third', annotation_position='top right')
    if n_days < 90:
        fig.add_vline(x=n_days + 0.5, line_dash='dash', line_color=GREY,
                      annotation_text='model window ends', annotation_position='top')
    return _layout(fig, height=360, xaxis_title='day', yaxis_title='usage',
                   legend=dict(orientation='h', y=-0.25))


def activity_heatmap(usage):
    """Services x days, cell = used that day or not."""
    act = (usage[DAY_COLS].values > 0).astype(int)
    labels = [SERVICE_LABELS.get(short_name(s), s) for s in usage.index]
    fig = go.Figure(go.Heatmap(
        z=act, x=np.arange(1, 91), y=labels, colorscale=[[0, '#f1f3f5'], [1, DOWN]],
        showscale=False, xgap=1, ygap=1,
        hovertemplate='%{y}, day %{x}<extra></extra>',
    ))
    return _layout(fig, height=380, xaxis_title='day')


def shape_function(d, name, lo=None, hi=None):
    """Step plot of one main-effect term from explain_global().data(i)."""
    x = np.asarray(d['names'], dtype=float)
    y = np.asarray(d['scores'], dtype=float)
    lower = np.asarray(d.get('lower_bounds', y), dtype=float)
    upper = np.asarray(d.get('upper_bounds', y), dtype=float)
    xs = x[:-1]
    if lo is not None and hi is not None:
        keep = (xs >= lo) & (xs <= hi)
        xs, y, lower, upper = xs[keep], y[keep], lower[keep], upper[keep]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=upper, mode='lines', line=dict(width=0, shape='hv'),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=xs, y=lower, mode='lines', line=dict(width=0, shape='hv'),
                             fill='tonexty', fillcolor='rgba(141,153,174,0.3)', showlegend=False,
                             hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=xs, y=y, mode='lines', line=dict(color=UP, width=2, shape='hv'),
                             name='score', hovertemplate='value %{x:.3g}<br>log-odds %{y:.3f}<extra></extra>'))
    fig.add_hline(y=0, line_dash='dash', line_color='black', line_width=1)
    return _layout(fig, height=380, title=pretty(name), xaxis_title='feature value',
                   yaxis_title='log-odds of churn', showlegend=False)


def interaction_heatmap(d, name):
    left = np.asarray(d['left_names'], dtype=float)
    right = np.asarray(d['right_names'], dtype=float)
    z = np.asarray(d['scores'], dtype=float)
    a, b = name.split(' & ')
    lim = np.abs(z).max() or 1
    fig = go.Figure(go.Heatmap(
        z=z.T, x=[f'{v:.3g}' for v in left[:-1]], y=[f'{v:.3g}' for v in right[:-1]],
        colorscale=[[0, DOWN], [0.5, '#ffffff'], [1, UP]], zmin=-lim, zmax=lim,
        colorbar=dict(title='log-odds'),
    ))
    return _layout(fig, height=420, title=pretty(name), xaxis_title=pretty(a), yaxis_title=pretty(b))


def importance_bar(imp, top=20):
    s = imp.head(top)[::-1]
    fig = go.Figure(go.Bar(
        x=s.values, y=[pretty(n) for n in s.index], orientation='h',
        marker_color=[GREY if ' & ' in n else UP for n in s.index],
        hovertemplate='%{y}<br>mean |contribution|: %{x:.3f}<extra></extra>',
    ))
    return _layout(fig, height=max(400, 26 * len(s) + 60),
                   xaxis_title='mean |log-odds contribution| (grey = interaction)')


def probability_histogram(df, thr):
    fig = go.Figure()
    for label, name, color in [(0, 'Stayed', DOWN), (1, 'Churned', UP)]:
        sub = df[df.churn == label] if 'churn' in df else df
        fig.add_trace(go.Histogram(x=sub.churn_proba, name=name, marker_color=color,
                                   opacity=0.75, xbins=dict(start=0, end=1, size=0.02)))
        if 'churn' not in df:
            break
    fig.add_vline(x=thr, line_dash='dash', line_color='black',
                  annotation_text=f'threshold {thr:.2f}')
    return _layout(fig, height=340, barmode='overlay', xaxis_title='predicted churn probability',
                   yaxis_title='customers', yaxis_type='log', legend=dict(orientation='h', y=1.12))


def risk_donut(df):
    counts = df.risk.value_counts().reindex(['High', 'Medium', 'Low']).fillna(0)
    fig = go.Figure(go.Pie(labels=counts.index, values=counts.values, hole=0.55,
                           marker_colors=[RISK_COLORS[r] for r in counts.index], sort=False))
    return _layout(fig, height=300, legend=dict(orientation='h', y=-0.1))
