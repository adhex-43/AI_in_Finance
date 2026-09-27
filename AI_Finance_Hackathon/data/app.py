import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

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

# ---------------- Data ----------------
@st.cache_data(show_spinner='Loading financial intelligence...')
def load_data():
    ts=pd.read_csv('train_signals.csv'); te=pd.read_csv('test_signals.csv')
    tt = pd.read_parquet(
    'AI_Finance_Hackathon/data/train_transactions.parquet'
)
    ts['signal_sanasi']=pd.to_datetime(ts['signal_sanasi']); te['signal_sanasi']=pd.to_datetime(te['signal_sanasi'])
    tet = pd.read_parquet(
    'AI_Finance_Hackathon/data/test_transactions.parquet'
)
    return ts,te,tt,tet

train_signals,test_signals,train_tx,test_tx=load_data()

tx=train_tx.merge(train_signals[['signal_id','signal_sanasi']],on='signal_id',how='left')
tx['days_before_signal']=(tx['signal_sanasi'].dt.normalize()-tx['tranzaksiya_vaqti'].dt.normalize()).dt.days
tx['hour']=tx['tranzaksiya_vaqti'].dt.hour
tx['weekday']=tx['tranzaksiya_vaqti'].dt.weekday
tx['is_weekend']=(tx['weekday']>=5).astype(int)
tx['is_night']=((tx['hour']<6)|(tx['hour']>=22)).astype(int)

N=len(train_signals); NT=len(train_tx); TEST=len(test_signals); RATE=train_signals.eskalatsiya.mean()*100; AVG=NT/N

# ---------------- Plot theme ----------------
def fig_theme(fig,h=350):
    fig.update_layout(height=h,paper_bgcolor='rgba(0,0,0,0)',plot_bgcolor='#0a101c',font=dict(family='Inter',color='#8b98aa'),margin=dict(l=20,r=20,t=48,b=30),title_font=dict(family='Space Grotesk',size=16,color='#edf7ff'),xaxis=dict(gridcolor='rgba(148,163,184,.07)',zerolinecolor='rgba(148,163,184,.07)'),yaxis=dict(gridcolor='rgba(148,163,184,.07)',zerolinecolor='rgba(148,163,184,.07)'),hoverlabel=dict(bgcolor='#101827',font_color='#fff'))
    return fig

# ---------------- Hero ----------------
st.markdown(f'''
<div class="hero"><div class="hero-content">
<div class="status"><span class="dot"></span>SYSTEM OPERATIONAL · TRAIN DATA INDEXED</div>
<div class="kicker">AI IN FINANCE / SIGNAL ANALYTICS</div>
<h1>Financial Signal<br>Intelligence</h1>
<p>Exploring historical transaction behavior to uncover the patterns behind financial signal escalation — from raw transactions to signal-level intelligence.</p>
<div class="mini-row"><div class="mini"><b>{N:,}</b><small>SIGNALS</small></div><div class="mini"><b>{NT/1e6:.2f}M</b><small>TRANSACTIONS</small></div><div class="mini"><b>180D</b><small>HISTORY WINDOW</small></div></div>
</div><div class="orbit"><div class="core"><div><strong>{RATE:.2f}%</strong><span>ESCALATION RATE</span></div></div></div></div>
''',unsafe_allow_html=True)

# ---------------- Navigation ----------------
if 'page' not in st.session_state: st.session_state.page='Overview'
nav=st.columns(6)
for i,(label,key) in enumerate([('◈ Overview','Overview'),('◎ Target','Target'),('⌁ Transactions','Transactions'),('◷ Temporal','Temporal'),('◇ Behavior','Behavior'),('✦ Findings','Findings')]):
    if nav[i].button(label,use_container_width=True): st.session_state.page=key
page=st.session_state.page

# ---------------- KPIs ----------------
if page=='Overview':
    st.markdown('<div class="section">System Overview</div><div class="sub">A high-density view of the financial signal ecosystem.</div>',unsafe_allow_html=True)
    cols=st.columns(5)
    vals=[('TRAIN SIGNALS',f'{N:,}','labeled observations'),('TRANSACTIONS',f'{NT:,}','historical records'),('ESCALATION',f'{RATE:.2f}%','positive class'),('AVG / SIGNAL',f'{AVG:.1f}','transactions'),('HIDDEN TEST',f'{TEST:,}','prediction set')]
    for c,(a,b,d) in zip(cols,vals): c.markdown(f'<div class="card"><div class="label">{a}</div><div class="value">{b}</div><div class="hint">{d}</div></div>',unsafe_allow_html=True)

    a,b=st.columns([1,1.35])
    with a:
        counts=train_signals.eskalatsiya.value_counts().sort_index()
        fig=go.Figure(go.Pie(labels=['Dismissed','Escalated'],values=[counts.get(0,0),counts.get(1,0)],hole=.75,marker=dict(colors=['#263348','#00e5ff'],line=dict(color='#0a101c',width=4)),textinfo='percent'))
        fig.update_layout(title='Escalation Distribution',annotations=[dict(text=f'{RATE:.1f}%',x=.5,y=.5,font=dict(size=28,color='#fff',family='Space Grotesk'),showarrow=False)])
        fig_theme(fig,365); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with b:
        monthly=train_signals.set_index('signal_sanasi').resample('ME').size().reset_index(name='signals')
        fig=go.Figure(go.Scatter(x=monthly.signal_sanasi,y=monthly.signals,mode='lines',line=dict(color='#00e5ff',width=3),fill='tozeroy',fillcolor='rgba(0,229,255,.07)'))
        fig.update_layout(title='Signal Generation Timeline'); fig_theme(fig,365); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

    st.markdown('<div class="section">Transaction Flow</div><div class="sub">How the transaction universe is distributed across direction and type.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        d=train_tx.kirim_chiqim.value_counts().reset_index(); d.columns=['direction','count']
        fig=px.bar(d,x='count',y='direction',orientation='h',title='Incoming vs outgoing',text='count'); fig.update_traces(marker_color='#00e5ff',texttemplate='%{text:,}',textposition='outside'); fig_theme(fig,350); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with b:
        d=train_tx.tranzaksiya_turi.value_counts().reset_index(); d.columns=['type','count']
        fig=px.bar(d,x='type',y='count',title='Transaction types',text='count'); fig.update_traces(marker_color='#8b5cf6',texttemplate='%{text:,}',textposition='outside'); fig_theme(fig,350); st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

elif page=='Target':
    st.markdown('<div class="section">Target Intelligence</div><div class="sub">The classification target the ML pipeline is learning to rank.</div>',unsafe_allow_html=True)
    counts=train_signals.eskalatsiya.value_counts().sort_index(); p0=counts.get(0,0)/N*100;p1=counts.get(1,0)/N*100
    fig=go.Figure(go.Bar(x=['Dismissed (0)','Escalated (1)'],y=[counts.get(0,0),counts.get(1,0)],marker_color=['#263348','#00e5ff'],text=[f'{counts.get(0,0):,}',f'{counts.get(1,0):,}'],textposition='outside')); fig.update_layout(title='Training Target Distribution');fig_theme(fig,430);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    cols=st.columns(3)
    for c,label,val in zip(cols,['DISMISSED','ESCALATED','POSITIVE RATE'],[f'{counts.get(0,0):,}',f'{counts.get(1,0):,}',f'{p1:.2f}%']): c.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{val}</div></div>',unsafe_allow_html=True)
    st.markdown('<div class="find"><b>Class imbalance is a core modeling consideration.</b><span>Only a minority of labeled signals were escalated. Ranking-oriented metrics such as ROC-AUC are therefore more informative than accuracy alone.</span></div>',unsafe_allow_html=True)

elif page=='Transactions':
    st.markdown('<div class="section">Transaction Intelligence</div><div class="sub">Decomposing nearly seven million historical records into interpretable patterns.</div>',unsafe_allow_html=True)
    a,b=st.columns(2)
    with a:
        d=train_tx.kirim_chiqim.value_counts().reset_index();d.columns=['direction','count'];fig=px.pie(d,names='direction',values='count',hole=.68,title='Transaction direction');fig_theme(fig,390);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    with b:
        d=train_tx.tranzaksiya_turi.value_counts().reset_index();d.columns=['type','count'];fig=px.bar(d,x='type',y='count',text='count',title='Transaction type');fig.update_traces(marker_color='#8b5cf6',texttemplate='%{text:,}',textposition='outside');fig_theme(fig,390);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    per=train_tx.groupby('signal_id').size().reset_index(name='transactions');fig=px.histogram(per,x='transactions',nbins=55,title='Transaction count distribution per signal');fig.update_traces(marker_color='#00e5ff');fig_theme(fig,390);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

elif page=='Temporal':
    st.markdown('<div class="section">Temporal Intelligence</div><div class="sub">The 180-day historical window surrounding every signal.</div>',unsafe_allow_html=True)
    fig=px.histogram(tx,x='days_before_signal',nbins=60,title='Transaction timing before signal');fig.update_traces(marker_color='#00e5ff');fig_theme(fig,400);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    a,b,c,d=st.columns(4); s=tx.days_before_signal.describe();
    for col,label,val in zip([a,b,c,d],['MIN','MEDIAN','75TH PERCENTILE','MAX'],[s['min'],s['50%'],s['75%'],s['max']]): col.markdown(f'<div class="card"><div class="label">{label}</div><div class="value">{val:.0f}<span style="font-size:12px;color:#58677c"> DAYS</span></div></div>',unsafe_allow_html=True)
    monthly=tx.assign(day=tx.signal_sanasi.dt.to_period('M').dt.to_timestamp()).groupby('day').size().reset_index(name='transactions');fig=px.line(monthly,x='day',y='transactions',title='Historical transaction volume by signal month');fig.update_traces(line_color='#8b5cf6');fig_theme(fig,350);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})

elif page=='Behavior':
    st.markdown('<div class="section">Behavioral Intelligence</div><div class="sub">Signal-level features derived from transaction timing and activity.</div>',unsafe_allow_html=True)
    behavior=tx.groupby('signal_id').agg(transaction_count=('signal_id','size'),last_1d=('days_before_signal',lambda x:(x<=1).sum()),last_7d=('days_before_signal',lambda x:(x<=7).sum()),last_30d=('days_before_signal',lambda x:(x<=30).sum()),weekend_ratio=('is_weekend','mean'),night_ratio=('is_night','mean')).reset_index().merge(train_signals[['signal_id','eskalatsiya']],on='signal_id')
    behavior['outcome']=behavior.eskalatsiya.map({0:'Dismissed',1:'Escalated'})
    metric=st.selectbox('Behavioral signal',['last_1d','last_7d','last_30d','weekend_ratio','night_ratio'])
    d=behavior.groupby('outcome')[metric].mean().reset_index();fig=px.bar(d,x='outcome',y=metric,text_auto='.3f',title=f'{metric} by outcome');fig.update_traces(marker_color='#00e5ff');fig_theme(fig,400);st.plotly_chart(fig,use_container_width=True,config={'displayModeBar':False})
    summary=behavior.groupby('outcome')[['transaction_count','last_1d','last_7d','last_30d','weekend_ratio','night_ratio']].mean().round(3);st.dataframe(summary,use_container_width=True)

elif page=='Findings':
    st.markdown('<div class="section">Key Findings</div><div class="sub">The main observations that informed signal-level feature engineering.</div>',unsafe_allow_html=True)
    findings=[('01','Imbalanced target',f'{RATE:.2f}% of training signals were escalated, making this an imbalanced binary classification problem.'),('02','Large transaction universe',f'{NT:,} historical transaction records are linked to {N:,} labeled signals.'),('03','180-day history', 'Transactions span from the signal day to 180 days before the signal.'),('04','Behavior is multi-dimensional','Direction, transaction type, volume, timing and amount statistics provide complementary signal-level descriptors.'),('05','Recent activity matters','1-day, 7-day and 30-day windows provide a way to capture changes close to signal generation.'),('06','Modeling unit is the signal','Millions of transaction rows are aggregated into one feature vector per signal before classification.')]
    for n,t,x in findings: st.markdown(f'<div class="find"><div style="color:#00e5ff;font-size:9px;font-weight:800;letter-spacing:2px">FINDING {n}</div><b>{t}</b><span>{x}</span></div>',unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;padding:35px 0;color:#59677b;font-size:11px;letter-spacing:2px">RAW TRANSACTIONS → BEHAVIORAL FEATURES → SIGNAL INTELLIGENCE</div>',unsafe_allow_html=True)
