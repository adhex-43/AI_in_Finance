import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os

# Fayllar har doim shu app.py turgan papkadan o'qiladi (Streamlit qayerdan ishga tushirsa ham)
BASE = os.path.dirname(os.path.abspath(__file__))
def P(name):
    return os.path.join(BASE, name)

st.set_page_config(page_title='AI in Finance — Signal Intelligence', page_icon='◈', layout='wide', initial_sidebar_state='collapsed')

# ---------------- CSS ----------------
st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');
:root{--bg:#050810;--panel:#0b1120;--line:rgba(148,163,184,.13);--muted:#718096;--text:#edf7ff;--cyan:#00e5ff;--violet:#8b5cf6;}
html,body,[class*="css"]{font-family:Inter,sans-serif}.stApp{background:radial-gradient(circle at 8% 0%,rgba(0,229,255,.08),transparent 28%),radial-gradient(circle at 95% 10%,rgba(139,92,246,.10),transparent 30%),var(--bg);color:var(--text)}
#MainMenu,footer,header{visibility:hidden}.block-container{max-width:1500px;padding:28px 38px 60px}
.hero{position:relative;overflow:hidden;min-height:405px;border:1px solid rgba(0,229,255,.15);border-radius:28px;padding:48px 52px;background:linear-gradient(120deg,rgba(11,17,32,.96),rgba(6,10,20,.93));box-shadow:0 30px 90px rgba(0,0,0,.42)}
.hero:after{content:"";position:absolute;inset:0;background-image:linear-gradient(rgba(0,229,255,.06) 1px,transparent 1px),linear-gradient(90deg,rgba(0,229,255,.06) 1px,transparent 1px);background-size:42px 42px;mask-image:linear-gradient(to right,black,transparent 75%);pointer-events:none}
.hero-content{position:relative;z-index:2;max-width:700px}.status{display:inline-flex;gap:9px;align-items:center;border:1px solid rgba(0,229,255,.22);background:rgba(0,229,255,.045);border-radius:999px;padding:8px 13px;color:#9befff;font-size:10px;font-weight:800;letter-spacing:2px}.dot{width:7px;height:7px;border-radius:50%;background:var(--cyan);box-shadow:0 0 15px var(--cyan)}
.kicker{margin-top:35px;color:var(--cyan);font-size:11px;font-weight:800;letter-spacing:4px}.hero h1{font:700 60px/1 Space Grotesk;margin:10px 0 0;letter-spacing:-3px;background:linear-gradient(100deg,#fff,#c5faff 45%,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent}.hero p{color:#8290a4;max-width:650px;line-height:1.75;font-size:14px;margin-top:20px}.orbit{position:absolute;right:95px;top:48px;width:310px;height:310px;border:1px solid rgba(0,229,255,.16);border-radius:50%;box-shadow:0 0 80px rgba(0,229,255,.04)}.orbit:before{content:"";position:absolute;inset:36px;border:1px dashed rgba(139,92,246,.3);border-radius:50%}.core{position:absolute;inset:93px;border-radius:50%;display:grid;place-items:center;text-align:center;background:radial-gradient(circle,rgba(0,229,255,.15),rgba(7,12,23,.98) 68%);border:1px solid rgba(0,229,255,.3);box-shadow:0 0 50px rgba(0,229,255,.11)}.core strong{font:700 30px Space Grotesk;color:#f5fbff}.core span{display:block;font-size:8px;color:#718096;letter-spacing:2px;margin-top:4px}.mini-row{display:flex;gap:10px;margin-top:28px}.mini{padding:11px 15px;border:1px solid var(--line);background:rgba(255,255,255,.035);border-radius:12px}.mini b{font:700 18px Space Grotesk}.mini small{display:block;color:#536176;font-size:8px;letter-spacing:1.5px;margin-top:3px}
.nav{display:flex;gap:8px;align-items:center;margin:18px 0}.navlabel{color:#526076;font-size:9px;font-weight:800;letter-spacing:2px;margin-right:5px}.section{font:600 23px Space Grotesk;margin:34px 0 5px}.sub{font-size:12px;color:#657287;margin-bottom:16px}.card{border:1px solid var(--line);background:linear-gradient(135deg,rgba(13,20,35,.86),rgba(8,13,23,.78));border-radius:18px;padding:20px;min-height:120px}.card .label{font-size:9px;color:#64748b;font-weight:800;letter-spacing:1.8px}.card .value{font:700 29px Space Grotesk;margin-top:10px}.card .hint{font-size:10px;color:#4f5d72;margin-top:7px}
.find{border:1px solid rgba(0,229,255,.10);background:linear-gradient(135deg,rgba(0,229,255,.045),rgba(139,92,246,.035));border-radius:15px;padding:18px;margin-bottom:10px}.find b{font:600 16px Space Grotesk}.find span{display:block;color:#7c8a9d;font-size:12px;line-height:1.65;margin-top:5px}.stButton>button{border:1px solid rgba(0,229,255,.18);background:rgba(0,229,255,.045);color:#a5f3fc;border-radius:10px}.stButton>button:hover{border-color:rgba(0,229,255,.5);color:#fff}
@media(max-width:950px){.block-container{padding:20px}.hero{padding:34px 28px}.hero h1{font-size:43px}.orbit{right:-130px;opacity:.25}}
</style>
''', unsafe_allow_html=True)

# Extra styles for new blocks
st.markdown('''<style>
.step{border:1px solid var(--line);background:rgba(255,255,255,.03);border-radius:16px;padding:18px;min-height:150px}
.step .n{color:var(--cyan);font-size:10px;font-weight:800;letter-spacing:2px}.step b{display:block;font:600 16px Space Grotesk;margin-top:8px}
.step span{display:block;color:#7c8a9d;font-size:12px;line-height:1.6;margin-top:6px}
.note{color:#7c8a9d;font-size:12px;line-height:1.6;margin:-4px 0 18px}
</style>''', unsafe_allow_html=True)

# ---------------- Results from the submitted notebook (Hakathon_final.ipynb) ----------------
MODEL_HISTORY = [("Logistic Regression (15 global features)", 0.566),
                 ("HistGradientBoosting (global features)", 0.599),
                 ("LightGBM + burst/baseline, type×direction features", 0.636)]
FINAL_AUC = 0.6363
DECILES = [0.092, 0.104, 0.113, 0.117, 0.136, 0.163, 0.194, 0.226, 0.276, 0.296]
TOP_FEATURES = [
    ("all_amt_min", 0.0627), ("base_amtmean_bank_otkazmasi_chiqim", 0.0574),
    ("all_amtmean_bank_otkazmasi_chiqim", 0.0436), ("base_amtmean_naqd_kirim", 0.0429),
    ("all_amtmean_naqd_kirim", 0.0352), ("base_amt_min", 0.0329),
    ("base_amtmean_karta_chiqim", 0.0211), ("all_amtmean_karta_chiqim", 0.0202),
    ("base_amtmean_bank_otkazmasi_kirim", 0.0172), ("all_amtmean_bank_otkazmasi_kirim", 0.0162),
    ("w30_amtmean_bank_otkazmasi_chiqim", 0.0157), ("w7_amtmean_karta_chiqim", 0.0106),
    ("all_amtmean_xalqaro_chiqim", 0.0098), ("all_amt_max", 0.0084), ("w7_night_share", 0.0079)]

# ---------------- Data (all heavy work cached once; raw transactions are not kept in memory) ----------------
@st.cache_data(show_spinner='Loading financial intelligence...')
def compute_all():
    """Statistikalar to'liq ma'lumotdan make_site_data.py orqali oldindan hisoblangan (site_data/ papkasi)."""
    import json
    def d(f):
        # site_data/ papkasida yoki to'g'ridan-to'g'ri app.py yonida bo'lishi mumkin
        for p in (P(os.path.join('site_data', f)), P(f)):
            if os.path.exists(p):
                return p
        st.error(f"'{f}' fayli topilmadi. site_data papkasidagi fayllarni app.py bilan bir papkaga yuklang.")
        st.stop()
    S = json.load(open(d('summary.json')))
    R = {k: S[k] for k in ['n_train', 'n_test', 'n_tx', 'n_test_tx', 'rate', 'future_tx']}
    R['counts'] = pd.Series({int(k): v for k, v in S['counts'].items()}).sort_index()
    R['dir_counts'] = pd.Series(S['dir_counts']).sort_values(ascending=False)
    R['type_counts'] = pd.Series(S['type_counts']).sort_values(ascending=False)
    R['days_desc'] = pd.Series(S['days_desc'])
    R['amt_bins'] = np.array(S['amt_bins'])
    R['amt_hist'] = {int(k): np.array(v) for k, v in S['amt_hist'].items()}
    R['monthly_signals'] = pd.read_csv(d('monthly_signals.csv'), parse_dates=['signal_sanasi'])
    R['type_share'] = pd.read_csv(d('type_share.csv'), index_col=0)
    R['dir_share'] = pd.read_csv(d('dir_share.csv'), index_col=0)
    for k, name in [('type_share', 'tranzaksiya_turi'), ('dir_share', 'kirim_chiqim')]:
        R[k].index = R[k].index.astype(int); R[k].columns.name = name
    daily = pd.read_csv(d('daily.csv')); daily.columns = [c if c == 'days' else int(c) for c in daily.columns]
    R['daily'] = daily
    R['daily_dir'] = pd.read_csv(d('daily_dir.csv'))
    R['amt_by_type'] = pd.read_csv(d('amt_by_type.csv'), index_col=0)
    R['amt_by_type'].index.name = 'tranzaksiya_turi'
    R['sig'] = pd.read_csv(d('signal_level.csv.gz'))
    return R

R = compute_all()
N, NT, TEST, RATE = R['n_train'], R['n_tx'], R['n_test'], R['rate']
AVG = NT / N
PLOT = {'displayModeBar': False}
C0, C1 = '#526179', '#00e5ff'

def fig_theme(fig, h=350):
    fig.update_layout(height=h, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#0a101c', font=dict(family='Inter', color='#8b98aa'),
                      margin=dict(l=20, r=20, t=48, b=30), title_font=dict(family='Space Grotesk', size=16, color='#edf7ff'),
                      xaxis=dict(gridcolor='rgba(148,163,184,.07)', zerolinecolor='rgba(148,163,184,.07)'),
                      yaxis=dict(gridcolor='rgba(148,163,184,.07)', zerolinecolor='rgba(148,163,184,.07)'),
                      hoverlabel=dict(bgcolor='#101827', font_color='#fff'), legend=dict(bgcolor='rgba(0,0,0,0)'))
    return fig

def show(fig, h=350):
    st.plotly_chart(fig_theme(fig, h), width='stretch', config=PLOT)

def section(title, sub):
    st.markdown(f'<div class="section">{title}</div><div class="sub">{sub}</div>', unsafe_allow_html=True)

def insight(title, text):
    st.markdown(f'<div class="find"><b>{title}</b><span>{text}</span></div>', unsafe_allow_html=True)

def card(col, label, value, hint=''):
    col.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{value}</div><div class="hint">{hint}</div></div>', unsafe_allow_html=True)


REPO_URL = 'https://github.com/adhex-43/AI_in_Finance'
NOTEBOOK_URL = 'https://github.com/adhex-43/AI_in_Finance/blob/main/AI_Finance_Hackathon/data/Hakathon_final.ipynb'
LINK_STYLE = 'display:inline-block;margin:0 6px 6px 0;padding:9px 15px;border:1px solid rgba(0,229,255,.25);border-radius:10px;background:rgba(0,229,255,.05);color:#a5f3fc;text-decoration:none;font-size:12px;font-weight:600'
LINKS_HTML = (f'<a href="{REPO_URL}" target="_blank" style="{LINK_STYLE}">⌥ GitHub repository</a>'
              f'<a href="{NOTEBOOK_URL}" target="_blank" style="{LINK_STYLE}">▤ Reproducible notebook</a>')

# ---------------- Hero ----------------
st.markdown(f'''
<div class="hero"><div class="hero-content">
<div class="status"><span class="dot"></span>TEAM A3783E69 · FINAL MODEL ROC-AUC {FINAL_AUC:.3f}</div>
<div class="kicker">AI IN FINANCE / SIGNAL ANALYTICS</div>
<h1>Financial Signal<br>Intelligence</h1>
<p>Exploring historical transaction behavior to uncover the patterns behind alert escalation, and turning those patterns into a model that ranks which alerts are most likely to be escalated.</p>
<div class="mini-row"><div class="mini"><b>{N:,}</b><small>TRAIN SIGNALS</small></div><div class="mini"><b>{NT/1e6:.2f}M</b><small>TRANSACTIONS</small></div><div class="mini"><b>248</b><small>FEATURES</small></div><div class="mini"><b>{FINAL_AUC:.3f}</b><small>OOF ROC-AUC</small></div></div>
<div style="margin-top:22px">{LINKS_HTML}</div>
</div><div class="orbit"><div class="core"><div><strong>{RATE:.2f}%</strong><span>ESCALATION RATE</span></div></div></div></div>
''', unsafe_allow_html=True)

# ---------------- Navigation ----------------
PAGES = [('◈ Overview', 'Overview'), ('◎ Target', 'Target'), ('⌁ Transactions', 'Transactions'), ('◷ Temporal', 'Temporal'),
         ('◇ Escalated vs Dismissed', 'Compare'), ('⚙ Features & Model', 'Model'), ('✦ Findings', 'Findings')]
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'
nav = st.columns(len(PAGES))
for i, (label, key) in enumerate(PAGES):
    if nav[i].button(label, width='stretch'):
        st.session_state.page = key
page = st.session_state.page

# ================= OVERVIEW =================
if page == 'Overview':
    section('Our Approach', 'From millions of raw transactions to one escalation probability per alert.')
    steps = [('01', 'Explore', 'Studied target balance, transaction types, directions, amounts and timing relative to the alert date.'),
             ('02', 'Aggregate', 'Each alert has ~500 transactions. We summarised them into one feature vector per signal_id.'),
             ('03', 'Contrast', 'Split history into windows (1 / 7 / 30 days vs a 31–180 day baseline) to compare recent behavior with the client\'s normal behavior.'),
             ('04', 'Model', 'LightGBM with 5-fold stratified CV × 3 seeds; out-of-fold ROC-AUC used for every decision.'),
             ('05', 'Predict', 'Test predictions averaged over 15 models, merged by signal_id and validated against all submission rules.')]
    cols = st.columns(5)
    for c, (n, t, x) in zip(cols, steps):
        c.markdown(f'<div class="step"><div class="n">STEP {n}</div><b>{t}</b><span>{x}</span></div>', unsafe_allow_html=True)

    section('Dataset Overview', 'Four files: signals with labels, test signals, and the transaction history linked to each signal.')
    cols = st.columns(5)
    for c, v in zip(cols, [('TRAIN SIGNALS', f'{N:,}', 'labeled alerts'), ('TRAIN TRANSACTIONS', f'{NT:,}', 'historical records'),
                           ('TEST SIGNALS', f'{TEST:,}', 'hidden labels'), ('TEST TRANSACTIONS', f"{R['n_test_tx']:,}", 'historical records'),
                           ('AVG / SIGNAL', f'{AVG:.0f}', 'transactions per alert')]):
        card(c, *v)
    st.write('')
    schema = pd.DataFrame({
        'File': ['train_signals.csv', 'train_signals.csv', 'train_signals.csv', 'train_transactions.parquet', 'train_transactions.parquet',
                 'train_transactions.parquet', 'train_transactions.parquet', 'train_transactions.parquet'],
        'Column': ['signal_id', 'signal_sanasi', 'eskalatsiya', 'signal_id', 'tranzaksiya_vaqti', 'kirim_chiqim', 'tranzaksiya_turi', 'miqdor_indeksi'],
        'Meaning': ['Unique alert ID', 'Alert date', 'Target: 1 = escalated, 0 = dismissed', 'Link to the alert (many rows per alert)',
                    'Transaction timestamp', 'Direction: kirim (in) / chiqim (out)', 'karta, bank_otkazmasi, naqd, xalqaro', 'Standardised transaction size']})
    st.dataframe(schema, width='stretch', hide_index=True)
    insight('Relational structure', 'Transactions are a one-to-many table: the modeling unit is the signal, so every feature is an aggregation over that signal\'s history. '
            f'We also verified there are {R["future_tx"]} transactions after the alert date, so there is no look-ahead leakage.')

# ================= TARGET =================
elif page == 'Target':
    section('Target Distribution', 'What the model has to rank.')
    counts = R['counts']
    a, b = st.columns([1, 1.35])
    with a:
        fig = go.Figure(go.Pie(labels=['Dismissed', 'Escalated'], values=[counts.get(0, 0), counts.get(1, 0)], hole=.75,
                               marker=dict(colors=['#263348', C1], line=dict(color='#0a101c', width=4)), textinfo='percent'))
        fig.update_layout(title='Escalated vs dismissed', annotations=[dict(text=f'{RATE:.1f}%', x=.5, y=.5, font=dict(size=28, color='#fff'), showarrow=False)])
        show(fig, 380)
    with b:
        m = R['monthly_signals']
        fig = go.Figure()
        fig.add_bar(x=m['signal_sanasi'], y=m['signals'], name='Signals', marker_color='#263348')
        fig.add_scatter(x=m['signal_sanasi'], y=m['rate'] * 100, name='Escalation %', yaxis='y2', line=dict(color=C1, width=3))
        fig.update_layout(title='Signals and escalation rate by month', yaxis2=dict(overlaying='y', side='right', title='%', showgrid=False))
        show(fig, 380)
    cols = st.columns(3)
    card(cols[0], 'DISMISSED', f'{counts.get(0, 0):,}'); card(cols[1], 'ESCALATED', f'{counts.get(1, 0):,}'); card(cols[2], 'POSITIVE RATE', f'{RATE:.2f}%')
    st.write('')
    insight('Imbalanced, but AUC does not need resampling',
            'Only about one in six alerts is escalated. ROC-AUC measures ranking quality, so we did not oversample or re-weight classes; we optimised ranking directly.')
    insight('Stable over time', 'Train and test cover the same period (2025-01 to 2026-12) and the monthly escalation rate has no strong trend, so a random stratified split is a fair validation scheme.')

# ================= TRANSACTIONS =================
elif page == 'Transactions':
    section('Transaction Composition', 'Direction, type and size of ~7 million historical transactions.')
    a, b = st.columns(2)
    with a:
        d = R['dir_counts'].reset_index(); d.columns = ['direction', 'count']
        fig = px.pie(d, names='direction', values='count', hole=.68, title='Direction', color_discrete_sequence=[C1, '#8b5cf6'])
        show(fig, 380)
    with b:
        d = R['type_counts'].reset_index(); d.columns = ['type', 'count']
        fig = px.bar(d, x='type', y='count', text='count', title='Transaction type')
        fig.update_traces(marker_color='#8b5cf6', texttemplate='%{text:,}', textposition='outside'); show(fig, 380)
    a, b = st.columns(2)
    with a:
        fig = px.histogram(R['sig'], x='transactions', nbins=60, title='Transactions per signal')
        fig.update_traces(marker_color=C1); show(fig, 360)
    with b:
        d = R['amt_by_type'].reset_index().melt(id_vars='tranzaksiya_turi', var_name='direction', value_name='median')
        fig = px.bar(d, x='tranzaksiya_turi', y='median', color='direction', barmode='group', title='Median size by type and direction',
                     color_discrete_sequence=[C1, '#8b5cf6'])
        show(fig, 360)
    insight('Incoming dominates, cards and bank transfers dominate',
            'About three quarters of transactions are incoming. Cash and international transfers are rare, but they have their own size profiles, '
            'which is why we built features for every type × direction combination instead of global averages only.')

# ================= TEMPORAL =================
elif page == 'Temporal':
    section('Activity Before the Signal', 'How transaction activity evolves across the 180-day window before each alert.')
    d = R['daily'].rename(columns={0: 'Dismissed', 1: 'Escalated'})
    fig = go.Figure()
    fig.add_scatter(x=d['days'], y=d['Dismissed'], name='Dismissed', line=dict(color=C0, width=2))
    fig.add_scatter(x=d['days'], y=d['Escalated'], name='Escalated', line=dict(color=C1, width=2))
    fig.update_layout(title='Average transactions per signal, by days before the alert', xaxis=dict(autorange='reversed', title='days before alert'))
    show(fig, 420)
    dd = R['daily_dir']
    fig = go.Figure()
    for col, color in [('kirim', C1), ('chiqim', '#8b5cf6')]:
        if col in dd:
            fig.add_scatter(x=dd['days'], y=dd[col], name=col, line=dict(color=color, width=2))
    fig.update_layout(title='Incoming vs outgoing activity before the alert', xaxis=dict(autorange='reversed', title='days before alert'))
    show(fig, 360)
    s = R['days_desc']; cols = st.columns(4)
    for col, label, val in zip(cols, ['MIN', 'MEDIAN', '75TH PERCENTILE', 'MAX'], [s['min'], s['50%'], s['75%'], s['max']]):
        card(col, label, f'{val:.0f} <span style="font-size:12px;color:#58677c">DAYS</span>')
    st.write('')
    insight('Alerts are triggered by bursts', 'Activity rises sharply in the last days before the alert for almost every signal. '
            'Because the burst appears for both classes, its existence alone does not separate them; what matters is how the burst compares to the client\'s own baseline. '
            'This led to our window features (1 / 7 / 30 days) and burst-vs-baseline contrast features.')

# ================= COMPARE =================
elif page == 'Compare':
    section('Escalated vs Dismissed', 'Where do the two groups actually differ?')
    sig = R['sig']
    a, b = st.columns(2)
    with a:
        ts_ = R['type_share'].T.reset_index().rename(columns={0: 'Dismissed', 1: 'Escalated'})
        fig = go.Figure([go.Bar(x=ts_['tranzaksiya_turi'], y=ts_['Dismissed'] * 100, name='Dismissed', marker_color=C0),
                         go.Bar(x=ts_['tranzaksiya_turi'], y=ts_['Escalated'] * 100, name='Escalated', marker_color=C1)])
        fig.update_layout(title='Transaction type share (%)', barmode='group'); show(fig, 360)
    with b:
        bins = R['amt_bins']; mid = (bins[1:] + bins[:-1]) / 2
        fig = go.Figure([go.Scatter(x=mid, y=R['amt_hist'][0], name='Dismissed', line=dict(color=C0, width=2)),
                         go.Scatter(x=mid, y=R['amt_hist'][1], name='Escalated', line=dict(color=C1, width=2))])
        fig.update_layout(title='Transaction size distribution (density)', xaxis_title='miqdor_indeksi'); show(fig, 360)

    metrics = {'Minimum transaction size': 'amt_min', 'Mean transaction size': 'amt_mean',
               'Mean size: bank transfer OUT': 'amtmean_bank_otkazmasi_chiqim', 'Mean size: cash IN': 'amtmean_naqd_kirim',
               'Mean size: card OUT': 'amtmean_karta_chiqim', 'Transactions in last 7 days': 'last_7d',
               'Share of history in last 7 days': 'share_last_7d', 'Total transactions': 'transactions', 'Night ratio': 'night_ratio'}
    metrics = {k: v for k, v in metrics.items() if v in sig.columns}
    choice = st.selectbox('Signal-level metric', list(metrics))
    col = metrics[choice]
    a, b = st.columns([1.4, 1])
    with a:
        lo, hi = sig[col].quantile([.01, .99])
        fig = px.box(sig[sig[col].between(lo, hi)], x='outcome', y=col, color='outcome', points=False,
                     color_discrete_map={'Dismissed': C0, 'Escalated': C1}, title=f'{choice} by outcome')
        show(fig, 400)
    with b:
        dec = pd.qcut(sig[col].rank(method='first'), 5, labels=['Q1 low', 'Q2', 'Q3', 'Q4', 'Q5 high'])
        rate = sig.groupby(dec, observed=True)['eskalatsiya'].mean().mul(100).reset_index()
        fig = px.bar(rate, x=col, y='eskalatsiya', title='Escalation % by quintile', text_auto='.1f')
        fig.update_traces(marker_color=C1); fig.update_layout(xaxis_title='', yaxis_title='%'); show(fig, 400)

    summary = sig.groupby('outcome')[list(metrics.values())].mean().T
    summary.index = list(metrics.keys())
    st.dataframe(summary.round(3), width='stretch')
    insight('Global shares are almost identical', 'Type and direction shares differ by only a fraction of a percentage point between the classes '
            '(cash 6.3% vs 6.5%, outgoing 24.3% vs 24.7%). Simple counts and shares are weak signals.')
    insight('Size within a channel is the strongest signal', 'Escalated alerts tend to have smaller transactions overall and a different size profile '
            'inside specific channels (bank-transfer outflows, cash inflows, card outflows). This is visible in the quintile charts above and it is exactly what the model relied on most.')

# ================= MODEL =================
elif page == 'Model':
    section('Feature Engineering', '248 signal-level features, every group motivated by an EDA observation.')
    groups = [('Global statistics', 'Count, mean, std, min, max, median of transaction size over the whole history.', 'Baseline descriptors.'),
              ('Type × direction', 'Share and mean size for each of the 8 combinations (e.g. naqd_kirim, bank_otkazmasi_chiqim).', 'Global shares were nearly identical between classes; size inside a channel was not.'),
              ('Time windows', 'The same statistics for the last 1, 7, 30 days and for the 31–180 day baseline.', 'Activity bursts right before the alert.'),
              ('Burst vs baseline', 'Activity rate ratio, size z-score and share differences between recent windows and the baseline.', 'The burst exists for everyone; its deviation from normal behavior matters.'),
              ('Velocity', 'Gaps between transactions in seconds, max transactions per hour and per day.', 'Bursts can be very fast.'),
              ('Pass-through', 'Share of outflows occurring within 1h / 24h after an inflow, including similar-size pairs.', 'Classic money-movement pattern in monitoring.'),
              ('Habits & calendar', 'Night and weekend ratios, history span, signal month and weekday.', 'Behavioral context.')]
    st.dataframe(pd.DataFrame(groups, columns=['Feature group', 'What it measures', 'EDA motivation']), width='stretch', hide_index=True)

    section('Model Progression', 'Out-of-fold ROC-AUC, 5-fold stratified CV.')
    a, b = st.columns([1, 1])
    with a:
        h = pd.DataFrame(MODEL_HISTORY, columns=['model', 'auc'])
        fig = px.bar(h, x='auc', y='model', orientation='h', text='auc', title='ROC-AUC by iteration')
        fig.update_traces(marker_color=[C0, '#8b5cf6', C1], texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(xaxis=dict(range=[0.5, 0.66]), yaxis_title=''); show(fig, 360)
    with b:
        fig = go.Figure(go.Bar(x=[f'D{i + 1}' for i in range(10)], y=[v * 100 for v in DECILES], marker_color=C1, text=[f'{v * 100:.1f}%' for v in DECILES], textposition='outside'))
        fig.add_hline(y=RATE, line_dash='dash', line_color='#8b98aa', annotation_text='average')
        fig.update_layout(title='Real escalation rate by predicted-risk decile', xaxis_title='D1 = lowest predicted risk', yaxis_title='%'); show(fig, 360)

    fi = pd.DataFrame(TOP_FEATURES, columns=['feature', 'importance']).iloc[::-1]
    fig = px.bar(fi, x='importance', y='feature', orientation='h', title='Top-15 features (LightGBM gain share)')
    fig.update_traces(marker_color=C1); fig.update_layout(yaxis_title=''); show(fig, 520)
    insight('Validation setup', 'LightGBM (learning rate 0.02, 15 leaves, feature/bagging fraction, L2 regularisation, early stopping). '
            '5 folds × 3 seeds = 15 models; test predictions are their average. A HistGradientBoosting blend was tested (0.629) but did not improve the out-of-fold score, so the final model is LightGBM only.')
    insight('What the lift chart means', 'Alerts in the top predicted decile are escalated about 3× as often as those in the bottom decile (29.6% vs 9.2%), '
            'so the model can meaningfully prioritise the review queue even though the overall signal is weak.')

# ================= FINDINGS =================
elif page == 'Findings':
    section('Key Findings', 'What we learned and how it shaped the model.')
    findings = [
        ('01', 'Imbalanced target', f'{RATE:.2f}% of training alerts were escalated. We optimised ranking (ROC-AUC) directly rather than accuracy.'),
        ('02', 'Every alert sits on a burst', 'Activity jumps in the days before almost every alert, for both classes. The burst itself is not discriminative.'),
        ('03', 'Shares barely differ', 'Transaction type and direction shares are nearly identical between escalated and dismissed alerts.'),
        ('04', 'Size inside a channel matters most', 'Minimum transaction size and mean size of bank-transfer outflows, cash inflows and card outflows were the top features.'),
        ('05', 'Baseline history is informative', 'Many top features come from the 31–180 day baseline, i.e. the client\'s normal behavior, not only from the burst.'),
        ('06', 'Richer features beat model tuning', 'Moving from global aggregates to type×direction and window features raised out-of-fold ROC-AUC from 0.599 to 0.636.')]
    for n, t, x in findings:
        st.markdown(f'<div class="find"><div style="color:#00e5ff;font-size:9px;font-weight:800;letter-spacing:2px">FINDING {n}</div><b>{t}</b><span>{x}</span></div>', unsafe_allow_html=True)
    section('Conclusion', 'Summary and limitations.')
    insight('Summary', f'Our final LightGBM model reaches an out-of-fold ROC-AUC of {FINAL_AUC:.3f} and ranks alerts so that the riskiest decile is escalated ~3× more often than the safest. '
            'The strongest evidence of escalation lies in the size profile of specific transaction channels, compared with the client\'s own history.')
    insight('Limitations & next steps', 'The data is synthetic and the separating signal is weak (AUC ≈ 0.64). Next steps: sequence models over raw transactions, '
            'finer burst-shape features and hyperparameter search with repeated CV.')
    section('Code & Reproducibility', 'Everything needed to reproduce the submission and this website.')
    st.markdown(f'<div>{LINKS_HTML}</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;padding:35px 0;color:#59677b;font-size:11px;letter-spacing:2px">RAW TRANSACTIONS → BEHAVIORAL FEATURES → SIGNAL INTELLIGENCE · TEAM A3783E69</div>', unsafe_allow_html=True)
