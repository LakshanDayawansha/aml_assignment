# Feature engineering, same logic as data_preprocess/preprocess.ipynb
import numpy as np
import pandas as pd

N_DAYS = 90
DAY_COLS = ['Day_' + str(i) for i in range(1, 91)]

SERVICES = [
    'usage_app_facebook_daily', 'usage_app_helakuru_daily', 'usage_app_other',
    'usage_app_tiktok_daily', 'usage_app_whatsapp_daily', 'usage_app_youtube_daily',
    'usage_pack_data', 'usage_pack_vas',
    'usage_voice_d2d_incoming', 'usage_voice_d2d_outgoing',
    'usage_voice_d2nd_outgoing', 'usage_voice_nd2d_incoming',
]


def short_name(s):
    return s.replace('usage_', '').replace('_daily', '')


# n_days < 90 uses only the first n_days; the trend compares the last third with the first third
def series_features(df, n_days=N_DAYS):
    k = n_days // 3
    X = df[DAY_COLS[:n_days]].to_numpy(dtype=np.float32)
    act = X > 0
    f = pd.DataFrame()
    f['ratio_last30_first30'] = (X[:, -k:].mean(axis=1) + 1) / (X[:, :k].mean(axis=1) + 1)
    f['active_days'] = act.sum(axis=1)
    last = np.where(act.any(axis=1), n_days - 1 - np.argmax(act[:, ::-1], axis=1), -1)
    f['days_since_active'] = n_days - 1 - last
    f['customer_id'] = df.customer_id.values
    f['usage_type'] = df.usage_type.values
    return f


def to_wide(df):
    df = df.copy()
    df['usage_type'] = df['usage_type'].map(short_name)
    w = df.pivot(index='customer_id', columns='usage_type')
    w.columns = [u + '__' + f for f, u in w.columns]
    return w


def add_overall_features(w, usage_df, n_days=N_DAYS):
    k = n_days // 3
    w = w.copy()
    for feat in ['ratio_last30_first30', 'active_days', 'days_since_active']:
        cols = [c for c in w.columns if c.endswith('__' + feat)]
        w['all__' + feat + '_min'] = w[cols].min(axis=1)
        w['all__' + feat + '_mean'] = w[cols].mean(axis=1)
        w['all__' + feat + '_max'] = w[cols].max(axis=1)

    days = usage_df.set_index('customer_id')[DAY_COLS[:n_days]]
    svc = usage_df.usage_type.map(short_name)

    tot = days.groupby(level=0).sum().loc[w.index].values
    w['all__total'] = tot.sum(axis=1)
    w['all__last30_share'] = tot[:, -k:].sum(axis=1) / (tot.sum(axis=1) + 1)

    # data = apps + data pack
    row_tot = days.sum(axis=1)
    data = row_tot[svc.str.startswith(('app_', 'pack_data')).values].groupby(level=0).sum()
    voice = row_tot[svc.str.startswith('voice_').values].groupby(level=0).sum()
    w['data_voice_ratio'] = data / (voice + 1)

    w['n_services_active_last7'] = (days.iloc[:, -7:] > 0).any(axis=1).groupby(level=0).sum()
    return w


def complete_usage(usage):
    """Check an uploaded raw usage table and add all-zero rows for missing services."""
    missing = [c for c in ['customer_id', 'usage_type'] + DAY_COLS if c not in usage.columns]
    if missing:
        raise ValueError('missing columns: ' + ', '.join(missing[:5]) + ('...' if len(missing) > 5 else ''))
    unknown = set(usage.usage_type) - set(SERVICES)
    if unknown:
        raise ValueError('unknown usage_type values: ' + ', '.join(sorted(unknown)))

    usage = usage[['customer_id', 'usage_type'] + DAY_COLS].copy()
    usage[DAY_COLS] = usage[DAY_COLS].fillna(0).clip(lower=0)
    usage = usage.groupby(['customer_id', 'usage_type'], as_index=False).sum()

    full = pd.MultiIndex.from_product([usage.customer_id.unique(), SERVICES], names=['customer_id', 'usage_type'])
    usage = usage.set_index(['customer_id', 'usage_type']).reindex(full, fill_value=0).reset_index()
    return usage


def build_features(usage, n_days=N_DAYS):
    """Raw daily usage (long format, one row per customer and service) -> one feature row per customer."""
    usage = complete_usage(usage)
    wide = to_wide(series_features(usage, n_days=n_days))
    return add_overall_features(wide, usage, n_days=n_days)
