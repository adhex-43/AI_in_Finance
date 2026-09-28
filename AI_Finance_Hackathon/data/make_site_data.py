"""
Sayt uchun statistikalarni to'liq ma'lumotdan bir marta hisoblaydi va site_data/ papkasiga saqlaydi.
Ishga tushirish: python make_site_data.py   (xom fayllar shu skript bilan bir papkada bo'lishi kerak)
"""
import os, json
import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
def P(name):
    return os.path.join(BASE, name)

def compute_all():
    import pyarrow.parquet as pq
    ts = pd.read_csv(P('train_signals.csv'), parse_dates=['signal_sanasi'])
    te = pd.read_csv(P('test_signals.csv'))
    n_test_tx = pq.ParquetFile(P('test_transactions.parquet')).metadata.num_rows
    tx = pd.read_parquet(P('train_transactions.parquet'))
    tx['tranzaksiya_vaqti'] = pd.to_datetime(tx['tranzaksiya_vaqti'])
    for c in ['kirim_chiqim', 'tranzaksiya_turi', 'signal_id']:
        tx[c] = tx[c].astype('category')
    tx = tx.merge(ts[['signal_id', 'signal_sanasi', 'eskalatsiya']], on='signal_id', how='left')
    tx['days'] = (tx['signal_sanasi'] - tx['tranzaksiya_vaqti'].dt.normalize()).dt.days.astype('int16')
    h = tx['tranzaksiya_vaqti'].dt.hour
    tx['night'] = ((h < 6) | (h >= 22)).astype('int8')
    tx['weekend'] = (tx['tranzaksiya_vaqti'].dt.weekday >= 5).astype('int8')
    for w in (1, 7, 30):
        tx[f'last_{w}d'] = (tx['days'] <= w).astype('int8')

    R = {'n_train': len(ts), 'n_test': len(te), 'n_tx': len(tx), 'n_test_tx': n_test_tx,
         'rate': ts['eskalatsiya'].mean() * 100, 'future_tx': int((tx['days'] < 0).sum())}
    R['counts'] = ts['eskalatsiya'].value_counts().sort_index()
    R['monthly_signals'] = ts.set_index('signal_sanasi').resample('ME').agg(
        signals=('eskalatsiya', 'size'), rate=('eskalatsiya', 'mean')).reset_index()
    R['dir_counts'] = tx['kirim_chiqim'].value_counts()
    R['type_counts'] = tx['tranzaksiya_turi'].value_counts()
    R['days_desc'] = tx['days'].describe()

    # Share by class
    R['type_share'] = pd.crosstab(tx['eskalatsiya'], tx['tranzaksiya_turi'], normalize='index')
    R['dir_share'] = pd.crosstab(tx['eskalatsiya'], tx['kirim_chiqim'], normalize='index')

    # Daily activity before the signal, per class (avg transactions per signal)
    per_class = R['counts']
    daily = tx.groupby(['eskalatsiya', 'days']).size().unstack(0)
    R['daily'] = daily.div(per_class, axis=1).reset_index()
    R['daily_dir'] = (tx.groupby(['days', 'kirim_chiqim'], observed=True).size().unstack() / R['n_train']).reset_index()

    # Amount distribution by class
    bins = np.linspace(-3, 7, 81)
    R['amt_bins'] = bins
    R['amt_hist'] = {c: np.histogram(tx.loc[tx['eskalatsiya'] == c, 'miqdor_indeksi'], bins=bins, density=True)[0] for c in (0, 1)}
    R['amt_by_type'] = tx.groupby(['tranzaksiya_turi', 'kirim_chiqim'], observed=True)['miqdor_indeksi'].median().unstack()

    # Signal-level table
    g = tx.groupby('signal_id', observed=True)
    sig = pd.DataFrame({
        'transactions': g.size(), 'last_1d': g['last_1d'].sum(), 'last_7d': g['last_7d'].sum(),
        'last_30d': g['last_30d'].sum(), 'night_ratio': g['night'].mean(), 'weekend_ratio': g['weekend'].mean(),
        'amt_mean': g['miqdor_indeksi'].mean(), 'amt_min': g['miqdor_indeksi'].min(), 'amt_max': g['miqdor_indeksi'].max()})
    sig['share_last_7d'] = sig['last_7d'] / sig['transactions']
    td = tx.groupby(['signal_id', 'tranzaksiya_turi', 'kirim_chiqim'], observed=True)['miqdor_indeksi'].mean().unstack([1, 2])
    td.columns = [f'amtmean_{a}_{b}' for a, b in td.columns]
    sig = sig.join(td).join(ts.set_index('signal_id')['eskalatsiya'])
    sig['outcome'] = sig['eskalatsiya'].map({0: 'Dismissed', 1: 'Escalated'})
    R['sig'] = sig.reset_index()
    return R



if __name__ == "__main__":
    R = compute_all()
    out = P("site_data"); os.makedirs(out, exist_ok=True)
    summary = {k: R[k] for k in ["n_train", "n_test", "n_tx", "n_test_tx", "rate", "future_tx"]}
    summary = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in summary.items()}
    summary["counts"] = {str(k): int(v) for k, v in R["counts"].items()}
    summary["dir_counts"] = {str(k): int(v) for k, v in R["dir_counts"].items()}
    summary["type_counts"] = {str(k): int(v) for k, v in R["type_counts"].items()}
    summary["days_desc"] = {str(k): float(v) for k, v in R["days_desc"].items()}
    summary["amt_bins"] = [float(x) for x in R["amt_bins"]]
    summary["amt_hist"] = {str(c): [float(x) for x in R["amt_hist"][c]] for c in (0, 1)}
    json.dump(summary, open(os.path.join(out, "summary.json"), "w"), indent=1)
    R["monthly_signals"].to_csv(os.path.join(out, "monthly_signals.csv"), index=False)
    R["type_share"].to_csv(os.path.join(out, "type_share.csv"))
    R["dir_share"].to_csv(os.path.join(out, "dir_share.csv"))
    R["daily"].to_csv(os.path.join(out, "daily.csv"), index=False)
    R["daily_dir"].to_csv(os.path.join(out, "daily_dir.csv"), index=False)
    R["amt_by_type"].to_csv(os.path.join(out, "amt_by_type.csv"))
    R["sig"].to_csv(os.path.join(out, "signal_level.csv.gz"), index=False, compression="gzip")
    print("Saqlandi:", sorted(os.listdir(out)))
