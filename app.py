import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
from itertools import combinations
# CatBoost modelini joblib ile yüklerken kütüphanenin ortamda bulunması gerekir
import catboost 

st.set_page_config(
    page_title="WC 2026 AI Predictor",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"],[data-testid="stMain"],
section[data-testid="stMain"],.stApp {background:#F5F5F7 !important;}
[data-testid="stHeader"]{background:#F5F5F7 !important;border-bottom:1px solid #E5E5EA;}
[data-testid="stSidebar"]{display:none;}
section[data-testid="stMain"]{padding-top:0 !important;}
[data-testid="stVerticalBlock"]{background:transparent;}
html,body,p,span,div,label{
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Display","SF Pro Text","Helvetica Neue",Roboto,sans-serif !important;
  color:#1D1D1F;
}

/* HERO */
.hero{background:#FFFFFF;border-bottom:1px solid #E5E5EA;padding:2.5rem 3rem 2rem;text-align:center;}
.hero-eye{font-size:.7rem;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:#86868B;margin-bottom:.4rem;}
.hero-title{font-size:2.8rem;font-weight:800;letter-spacing:-1.5px;color:#1D1D1F;margin:0 0 .3rem;}
.hero-sub{font-size:1rem;color:#86868B;margin:0;}

/* TABS */
[data-testid="stTabs"] [role="tablist"]{background:#FFFFFF !important;border-bottom:1px solid #E5E5EA !important;padding:0 2rem !important;gap:0 !important; justify-content: center;}
[data-testid="stTabs"] button[role="tab"]{background:transparent !important;color:#86868B !important;font-weight:600 !important;font-size:.95rem !important;border-radius:0 !important;border-bottom:2px solid transparent !important;padding:.8rem 1.5rem !important;transition:all .2s !important;}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{color:#1D1D1F !important;border-bottom:2px solid #1D1D1F !important;background:transparent !important;}
[data-testid="stTabs"] [role="tabpanel"]{padding-top:2rem;background:#F5F5F7 !important;}

/* SELECTBOX */
[data-testid="stSelectbox"]>div>div{background:#FFFFFF !important;border:1px solid #D2D2D7 !important;border-radius:12px !important;color:#1D1D1F !important;font-size:.95rem !important;}
[data-testid="stSelectbox"] label p{color:#86868B !important;font-size:.75rem !important;font-weight:700 !important;letter-spacing:.5px !important;text-transform:uppercase !important; text-align: center; width: 100%;}

/* BUTTONS */
[data-testid="stButton"] { display: flex !important; justify-content: center !important; width: 100% !important; }
[data-testid="stButton"] > button {
    border-radius: 980px !important; font-size: .95rem !important; font-weight: 600 !important; 
    padding: .7rem 1.5rem !important; width: 100% !important; max-width: 320px !important; 
    margin: 0 auto !important; transition: all .2s !important; box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
    display: block !important;
}
/* Primary Button (Koyu Antrasit / Endüstriyel) */
[data-testid="stButton"] > button[kind="primary"] { background: #FF3B30 !important; color: #FFF !important; border: none !important; }
[data-testid="stButton"] > button[kind="primary"]:hover { background: #D70015 !important; transform: translateY(-1px); box-shadow: 0 4px 12px rgba(28,28,30,0.2) !important; }
/* Secondary Button */
[data-testid="stButton"] > button[kind="secondary"] { background: #FFFFFF !important; color: #1D1D1F !important; border: 1px solid #D2D2D7 !important; }
[data-testid="stButton"] > button[kind="secondary"]:hover { background: #F5F5F7 !important; transform: translateY(-1px); border-color: #86868B !important; }
[data-testid="stButton"] > button:disabled { background: #E5E5EA !important; color: #A1A1A6 !important; cursor: not-allowed; transform: none !important; box-shadow: none !important; border: none !important; }

/* RADIO */
[data-testid="stRadio"] {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important; /* Dış kutuyu tam merkeze kilitler */
  width: 100% !important;
  margin-bottom: 1rem !important;
}
[data-testid="stRadio"] label p {
  color: #1D1D1F !important;
  font-size: .85rem !important;
  font-weight: 600 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] {
  display: flex !important;
  justify-content: center !important; /* Butonları kendi içinde ortalar */
  align-items: center !important;
  width: 100% !important;
  gap: 3rem !important; /* İki buton arasındaki boşluğu ferahlatır */
}

/* CENTER CONTAINER */
.block-container{max-width:1200px !important;margin:0 auto !important;padding-left:2rem !important;padding-right:2rem !important;}

/* SECTION LABEL */
.sec-lbl{font-size:.7rem;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:#86868B;margin-bottom:.75rem;text-align:center;}

/* BUTTON HINT */
.btn-hint{display:block;text-align:center;color:#86868B;font-size:.76rem;font-weight:500;margin-bottom:.5rem;letter-spacing:.3px;}

/* APPLE CARD */
.a-card{background:#FFFFFF;border-radius:18px;box-shadow:0 2px 16px rgba(0,0,0,.07);padding:1.5rem 1.75rem;margin-bottom:1rem;}

/* VS & TEAM */
.t-block{display:flex;align-items:center;gap:.75rem;background:#F5F5F7;border-radius:14px;border:1px solid #E5E5EA;padding:1rem 1.25rem;}
.t-flag{font-size:2.4rem;line-height:1;}
.t-name{font-size:1.1rem;font-weight:700;color:#1D1D1F;}
.t-sub{font-size:.75rem;color:#86868B;margin-top:2px;}
.vs-mid{text-align:center;font-weight:800;color:#D2D2D7;font-size:1.2rem;padding-top:1.3rem;}

/* PROB BAR */
.p-bar-wrap{margin:1.25rem 0 .25rem;}
.p-bar-labels{display:flex;justify-content:space-between;margin-bottom:.5rem;}
.p-bar-lbl{font-size:.72rem;font-weight:700;text-transform:uppercase;letter-spacing:.5px;}
.p-bar{display:flex;height:12px;border-radius:999px;overflow:hidden;background:#F5F5F7;gap:3px;}
.p-seg-h{background:linear-gradient(90deg,#004FBD,#0071E3);border-radius:999px 0 0 999px;}
.p-seg-a{background:linear-gradient(90deg,#C00,#FF3B30);border-radius:0 999px 999px 0;margin-left:auto;}
.p-nums{display:flex;justify-content:space-between;margin-top:.75rem;}
.p-pct-h{font-size:2.4rem;font-weight:800;color:#0071E3;letter-spacing:-1.5px;}
.p-pct-a{font-size:2.4rem;font-weight:800;color:#FF3B30;letter-spacing:-1.5px;text-align:right;}
.p-note{font-size:.72rem;color:#86868B;margin-top:2px;}

/* VERDICT */
.verdict{margin-top:1rem;padding:1rem 1.25rem;border-radius:12px;display:flex;align-items:center;justify-content:center;gap:.75rem;font-size:.95rem;font-weight:600;}
.v-h{background:#EBF3FF;border:1px solid #B3D4FF;color:#004FBD;}
.v-a{background:#FFF0EE;border:1px solid #FFD0CC;color:#C00;}
.v-e{background:#F5F5F7;border:1px solid #E5E5EA;color:#86868B;}

/* STATS */
.stat-row{display:flex;justify-content:space-between;align-items:center;padding:.6rem 0;border-bottom:1px solid #F5F5F7;}
.stat-row:last-child{border-bottom:none;}
.sv{font-size:.92rem;font-weight:700;min-width:80px;color:#1D1D1F;}
.sv-a{text-align:right;}
.sv-w{color:#30D158 !important;}
.sk{font-size:.8rem;color:#86868B;text-align:center;flex:1;}

/* H2H */
.h2h-row{display:flex;align-items:center;justify-content:space-between;padding:.55rem 0;border-bottom:1px solid #F5F5F7;font-size:.85rem;}
.h2h-row:last-child{border-bottom:none;}
.h2h-th{font-weight:600;color:#1D1D1F;flex:1;}
.h2h-ta{font-weight:600;color:#1D1D1F;flex:1;text-align:right;}
.h2h-sc{font-weight:800;color:#0071E3;background:#EBF3FF;padding:.15rem .6rem;border-radius:8px;font-size:.82rem;margin:0 .5rem;}
.h2h-yr{font-size:.72rem;color:#86868B;min-width:40px;text-align:center;}

/* FORM DOTS */
.fd{display:inline-block;width:26px;height:26px;border-radius:50%;font-size:.62rem;font-weight:800;text-align:center;line-height:26px;margin:0 2px;}
.fd-W{background:#30D158;color:white;}
.fd-D{background:#FF9500;color:white;}
.fd-L{background:#FF3B30;color:white;}

/* GROUP CARDS */
.grp-card{background:#FFFFFF;border-radius:12px;overflow:hidden;margin-bottom:1rem;box-shadow:0 2px 8px rgba(0,0,0,.04); border: 1px solid #E5E5EA;}
.grp-hdr{background:#F5F5F7;border-bottom:1px solid #E5E5EA;padding:.6rem .85rem;font-size:.7rem;font-weight:800;letter-spacing:2.5px;color:#1D1D1F;text-transform:uppercase; text-align: center;}
.grp-row{display:flex;align-items:center;gap:.45rem;padding:.5rem .85rem;border-bottom:1px solid #F5F5F7;border-left:3px solid transparent;}
.grp-row:last-child{border-bottom:none;}
.grp-pos{font-size:.68rem;font-weight:700;color:#86868B;width:14px;}
.grp-flag{font-size:1.15rem;}
.grp-name{font-size:.8rem;font-weight:600;color:#1D1D1F;flex:1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.grp-stat{font-size:.75rem;color:#86868B;width:18px;text-align:center;}
.grp-pts{font-size:.85rem;font-weight:700;color:#0071E3;width:22px;text-align:right;}
.grp-col-hdr{display:flex;padding:.4rem .85rem;font-size:.65rem;font-weight:700;color:#86868B;text-transform:uppercase;letter-spacing:.5px;border-bottom:1px solid #F5F5F7;background:#FAFAFA;}
.q-g{border-left-color:#30D158 !important; background: #F9FFF9;}
.q-o{border-left-color:#FF9500 !important; background: #FFFDF7;}

/* BRACKET CARDS */
.bkt-rnd{font-size:.75rem;font-weight:800;letter-spacing:2px;color:#86868B;text-transform:uppercase;text-align:center;margin:1rem 0 .5rem;}
.bkt-match{background:#FFFFFF;border-radius:12px;border:1px solid #E5E5EA;overflow:hidden;margin-bottom:.6rem;box-shadow:0 2px 6px rgba(0,0,0,.04);}
.bkt-team{display:flex;align-items:center;gap:.5rem;padding:.5rem .75rem;border-bottom:1px solid #F5F5F7;}
.bkt-team:last-child{border-bottom:none;}
.bkt-fl{font-size:1.1rem;}
.bkt-nm{font-size:.8rem;font-weight:600;color:#1D1D1F;flex:1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
.bkt-win{background:#F0F7FF !important;}
.bkt-win .bkt-nm{color:#0071E3 !important;font-weight:700;}
.bkt-tbd{color:#D2D2D7;font-style:italic;font-size:.75rem;}

/* CHAMPION */
.champ-wrap{background:linear-gradient(135deg,#FFFBEB,#FFF3CC);border:2px solid #FFD60A;border-radius:22px;padding:2.5rem 3rem;text-align:center;margin:2rem auto;max-width:500px; box-shadow: 0 8px 24px rgba(255,214,10,0.15);}
.champ-trophy{font-size:4.5rem;line-height:1;margin-bottom:.5rem;}
.champ-lbl{font-size:.75rem;font-weight:800;letter-spacing:3px;text-transform:uppercase;color:#86868B;}
.champ-fl{font-size:4rem;margin:.5rem 0;}
.champ-nm{font-size:2.8rem;font-weight:800;color:#1D1D1F;letter-spacing:-1px;}

/* BANNERS */
.b-ok{background:#EDFAF1;border:1px solid #A3E6B8;border-radius:12px;padding:1rem 1.25rem;font-size:.9rem;color:#1A7F3C;font-weight:600;text-align:center;margin-bottom:1.5rem;}
.b-warn{background:#FFF8E6;border:1px solid #FFD60A;border-radius:12px;padding:1rem 1.25rem;font-size:.9rem;color:#86520A;text-align:center;margin-bottom:1.5rem;font-weight:500;}
.b-info{background:#EBF3FF;border:1px solid #B3D4FF;border-radius:12px;padding:1rem 1.25rem;font-size:.9rem;color:#004FBD;text-align:center;margin-bottom:1rem;font-weight:500;}

/* PROGRESS */
.prog-bar{height:8px;border-radius:999px;background:#E5E5EA;margin:.5rem 0 1rem;}
.prog-fill{height:8px;border-radius:999px;background:#0071E3;transition:width .4s ease-in-out;}

/* DISCLAIMER */
.disclaimer{font-size:.75rem;color:#86868B;line-height:1.65;padding:1.5rem 2rem;border-top:1px solid #E5E5EA;margin-top:3rem;text-align:center; max-width: 900px; margin-left: auto; margin-right: auto;}

/* DIVIDER */
.hr{height:1px;background:#E5E5EA;margin:2rem 0;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
FLAGS = {
    "Algeria":"🇩🇿","Argentina":"🇦🇷","Australia":"🇦🇺","Austria":"🇦🇹",
    "Belgium":"🇧🇪","Bosnia-Herzegovina":"🇧🇦","Brazil":"🇧🇷","Canada":"🇨🇦",
    "Cape Verde":"🇨🇻","Colombia":"🇨🇴","Croatia":"🇭🇷","Curaçao":"🇨🇼",
    "Czechia":"🇨🇿","DR Congo":"🇨🇩","Ecuador":"🇪🇨","Egypt":"🇪🇬",
    "England":"🏴󠁧󠁢󠁥󠁮󠁧󠁿","France":"🇫🇷","Germany":"🇩🇪","Ghana":"🇬🇭",
    "Haiti":"🇭🇹","Iran":"🇮🇷","Iraq":"🇮🇶","Ivory Coast":"🇨🇮",
    "Japan":"🇯🇵","Jordan":"🇯🇴","Mexico":"🇲🇽","Morocco":"🇲🇦",
    "Netherlands":"🇳🇱","New Zealand":"🇳🇿","Norway":"🇳🇴","Panama":"🇵🇦",
    "Paraguay":"🇵🇾","Portugal":"🇵🇹","Qatar":"🇶🇦","Saudi Arabia":"🇸🇦",
    "Scotland":"🏴󠁧󠁢󠁳󠁣󠁴󠁿","Senegal":"🇸🇳","South Africa":"🇿🇦",
    "South Korea":"🇰🇷","Spain":"🇪🇸","Sweden":"🇸🇪","Switzerland":"🇨🇭",
    "Tunisia":"🇹🇳","Türkiye":"🇹🇷","United States":"🇺🇸","Uruguay":"🇺🇾",
    "Uzbekistan":"🇺🇿",
}
FORM_ALIAS    = {"Bosnia-Herzegovina":"Bosnia and Herzegovina","Czechia":"Czech Republic","Türkiye":"Turkey"}
STATS_ALIAS   = {"Czechia":"Czech Republic","Türkiye":"Turkey"}
RESULTS_ALIAS = {"Türkiye":"Turkey","Czechia":"Czech Republic","Bosnia-Herzegovina":"Bosnia and Herzegovina"}

def flag(t): return FLAGS.get(t,"🏳️")

# ── GERÇEK FIFA DÜNYA KUPASI 2026 GRUPLARI (48 Takım · 12 Grup) ──────────────
WC_GROUPS = {
    "A": ["Mexico","South Korea","Czechia","South Africa"],
    "B": ["Switzerland","Canada","Qatar","Bosnia-Herzegovina"],
    "C": ["Brazil","Morocco","Scotland","Haiti"],
    "D": ["United States","Türkiye","Australia","Paraguay"],
    "E": ["Germany","Ecuador","Ivory Coast","Curaçao"],
    "F": ["Netherlands","Japan","Sweden","Tunisia"],
    "G": ["Belgium","Iran","Egypt","New Zealand"],
    "H": ["Spain","Uruguay","Saudi Arabia","Cape Verde"],
    "I": ["France","Senegal","Norway","Iraq"],
    "J": ["Argentina","Austria","Algeria","Jordan"],
    "K": ["Portugal","Colombia","DR Congo","Uzbekistan"],
    "L": ["England","Croatia","Panama","Ghana"],
}
TOTAL_GROUP_MATCHES = sum(len(list(combinations(v, 2))) for v in WC_GROUPS.values())  # 72

# ── ASSETS ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_assets():
    model  = joblib.load("football_production_model.pkl")
    df_fd  = pd.read_csv("final_football_data.csv")
    df_st  = pd.read_csv("wc_team_alltime_stats.csv").set_index("team")
    # Dosya eksik olma ihtimaline karşı try-except
    try:
        df_ft  = pd.read_csv("wc_prediction_features_2026.csv").set_index("team")
    except:
        df_ft = pd.DataFrame() 
    df_res = pd.read_csv("results.csv", parse_dates=["date"])
    return model, df_fd, df_st, df_ft, df_res

try:
    model, df_fd, df_stats, df_feats, df_results = load_assets()
    TEAMS = sorted(list(set(df_fd["home_team"].tolist() + df_fd["away_team"].tolist() + list(FLAGS.keys()))))
except Exception as e:
    st.error(f"Dosyalar yüklenemedi: {e}")
    st.stop()

# ── PREDICTION HELPERS ────────────────────────────────────────────────────────
def form_stats(team):
    lu = FORM_ALIAS.get(team, team)
    rows = df_fd[(df_fd["home_team"]==lu)|(df_fd["away_team"]==lu)].tail(1)
    if rows.empty: return 0.5,1.0,1.0
    r = rows.iloc[0]
    if r["home_team"]==lu: return r["home_form"],r["home_avg_scored"],r["home_avg_conceded"]
    return r["away_form"],r["away_avg_scored"],r["away_avg_conceded"]

def wc_stats(team):
    lu = STATS_ALIAS.get(team, team)
    try:
        r = df_stats.loc[lu]
        return float(r["total_wc_appearances"]),float(r["titles"]),float(r["finals_reached"])
    except: return 0.0,0.0,0.0

def feat_val(team,col,default=0.0):
    try: return float(df_feats.loc[team,col])
    except: return default

def predict_match(home,away):
    hf,hs,hc = form_stats(home); af,as_,ac = form_stats(away)
    ha,ht,hfi = wc_stats(home); aa,at,afi = wc_stats(away)
    row = pd.DataFrame([{"home_form":hf,"away_form":af,
        "home_avg_scored":hs,"away_avg_scored":as_,
        "home_avg_conceded":hc,"away_avg_conceded":ac,
        "home_wc_appearances":ha,"away_wc_appearances":aa,
        "home_titles":ht,"away_titles":at,
        "home_finals_reached":hfi,"away_finals_reached":afi}])
    p = model.predict_proba(row)[0]
    return p[0],p[1]  # p[0]=ev sahibi kazanır, p[1]=deplasman/beraberlik

def simulate_group(teams):
    stats = {t:{"P":0,"W":0,"L":0,"Pts":0} for t in teams}
    results = []
    for h,a in combinations(teams,2):
        hp,ap = predict_match(h,a)
        w = h if hp>ap else a; l = a if w==h else h
        stats[w]["W"]+=1; stats[w]["Pts"]+=3; stats[l]["L"]+=1
        stats[h]["P"]+=1; stats[a]["P"]+=1
        results.append((h,a,w))
    st_ = sorted(teams,key=lambda t:(-stats[t]["Pts"],-stats[t]["W"],-wc_stats(t)[0]))
    return st_,stats,results

def simulate_group_from_picks(teams,picks):
    stats = {t:{"P":0,"W":0,"L":0,"Pts":0} for t in teams}
    results = []
    for h,a in combinations(teams,2):
        w = picks.get(f"{h}_vs_{a}")
        if not w: continue
        l = a if w==h else h
        stats[w]["W"]+=1; stats[w]["Pts"]+=3; stats[l]["L"]+=1
        stats[h]["P"]+=1; stats[a]["P"]+=1
        results.append((h,a,w))
    st_ = sorted(teams,key=lambda t:(-stats[t]["Pts"],-stats[t]["W"],-wc_stats(t)[0]))
    return st_,stats,results

def get_best_thirds(standings,stats,n=8):
    thirds = []
    for g,teams in standings.items():
        if len(teams)>=3:
            t = teams[2]
            thirds.append((t,stats[g].get(t,{}).get("Pts",0),wc_stats(t)[0]))
    return [x[0] for x in sorted(thirds,key=lambda x:(-x[1],-x[2]))[:n]]

def build_r32(standings,best_thirds):
    """16 maçlık R32 bracket. 4 sütun × 4 maç."""
    def top(g,r): return (standings.get(g,["TBD"]*4)+["TBD"]*4)[r]
    bt = (best_thirds+["TBD"]*8)[:8]
    return [
        (top("A",0), top("B",1)), (top("C",0), top("D",1)),
        (top("E",0), top("F",1)), (top("G",0), top("H",1)),
        (top("I",0), top("J",1)), (top("K",0), top("L",1)),
        (top("B",0), top("A",1)), (top("D",0), top("C",1)),
        (top("F",0), top("E",1)), (top("H",0), top("G",1)),
        (top("J",0), top("I",1)), (top("L",0), top("K",1)),
        (bt[0], bt[1]), (bt[2], bt[3]),
        (bt[4], bt[5]), (bt[6], bt[7]),
    ]

# ── H2H & FORM ────────────────────────────────────────────────────────────────
def resolve_res(t): return RESULTS_ALIAS.get(t,t)

def get_h2h(ta,tb,n=5):
    ra,rb = resolve_res(ta),resolve_res(tb)
    mask = (((df_results["home_team"]==ra)&(df_results["away_team"]==rb))|
            ((df_results["home_team"]==rb)&(df_results["away_team"]==ra)))
    return df_results[mask].dropna(subset=["home_score","away_score"]).tail(n)

def get_form(team,n=5):
    lu = resolve_res(team)
    rows = df_results[(df_results["home_team"]==lu)|(df_results["away_team"]==lu)].dropna(subset=["home_score","away_score"]).tail(n)
    out=[]
    for _,r in rows.iterrows():
        hs,aws = float(r["home_score"]),float(r["away_score"])
        if r["home_team"]==lu: out.append("W" if hs>aws else("D" if hs==aws else"L"))
        else: out.append("W" if aws>hs else("D" if aws==hs else"L"))
    return out

# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_ss():
    for k,v in {"sim_mode":"AI","group_stage_done":False,"group_standings":{},
                "group_stats":{},"group_matches":{},"r32_bracket":[],
                "ko_results":{},"champion":None,
                "t1_result":None}.items():
        if k not in st.session_state:
            st.session_state[k] = v
init_ss()

def reset_tour():
    for k in ["group_stage_done","group_standings","group_stats","group_matches","r32_bracket","ko_results","champion"]:
        st.session_state[k] = ({} if k in("group_standings","group_stats","group_matches","ko_results")
                                else (False if k=="group_stage_done" else ([] if k=="r32_bracket" else None)))
    for k in [x for x in st.session_state if x.startswith("up_")]: del st.session_state[k]

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-eye">FIFA Dünya Kupası 2026 · 48 Takım · AI Powered</div>
  <div class="hero-title">Tahmin Motoru</div>
  <p class="hero-sub">CatBoost · %77.30 doğruluk · Form + WC Tarih + İstatistik</p>
</div>""", unsafe_allow_html=True)

tab_match, tab_sim = st.tabs(["⚽  Tekli Maç Tahmini","🏆  Turnuva Simülatörü"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — TEKLİ MAÇ TAHMİNİ
# ══════════════════════════════════════════════════════════════════════════════
with tab_match:
    _,col_m,_ = st.columns([1,5,1])
    with col_m:
        c1,cVS,c2 = st.columns([5,1,5])
        with c1:
            home = st.selectbox("Ev Sahibi Takım",TEAMS,
                index=TEAMS.index("France") if "France" in TEAMS else 0,
                format_func=lambda t:f"{flag(t)}  {t}",key="t1_home")
        with cVS: st.markdown('<div class="vs-mid">VS</div>',unsafe_allow_html=True)
        with c2:
            away = st.selectbox("Deplasman Takımı",TEAMS,
                index=TEAMS.index("Germany") if "Germany" in TEAMS else 1,
                format_func=lambda t:f"{flag(t)}  {t}",key="t1_away")
        _,bc,_ = st.columns([2,3,2])
        with bc:
            st.markdown('<span class="btn-hint">İki takım seçtikten sonra buraya tıklayın</span>',unsafe_allow_html=True)
            clicked = st.button("Tahmini Göster", key="t1_btn", type="primary", use_container_width=True)

        if clicked:
            if home==away:
                st.warning("Lütfen iki farklı takım seçin.")
            else:
                hp,ap = predict_match(home,away)
                st.session_state.t1_result = (home,away,hp,ap)

        if st.session_state.t1_result:
            home,away,hp,ap = st.session_state.t1_result
            hpct,apct = hp*100,ap*100
            st.markdown("<br>",unsafe_allow_html=True)

            _,rb,_ = st.columns([2,3,2])
            with rb:
                st.markdown('<span class="btn-hint">Yeni bir tahmin yapmak için sıfırlayın</span>',unsafe_allow_html=True)
                if st.button("✕  Tahmini Sıfırla", key="t1_reset", type="secondary", use_container_width=True):
                    st.session_state.t1_result = None
                    st.rerun()

            st.markdown("<br>",unsafe_allow_html=True)

            h_elo=int(feat_val(home,"elo_rating_2026")); a_elo=int(feat_val(away,"elo_rating_2026"))
            h_rank=int(feat_val(home,"fifa_rank_apr2026")); a_rank=int(feat_val(away,"fifa_rank_apr2026"))
            tc1,tc2,tc3 = st.columns([5,1,5])
            with tc1:
                st.markdown(f'<div class="t-block"><span class="t-flag">{flag(home)}</span><div><div class="t-name">{home}</div><div class="t-sub">ELO {h_elo} · FIFA #{h_rank}</div></div></div>',unsafe_allow_html=True)
            with tc2:
                st.markdown('<div style="text-align:center;padding-top:1.3rem;color:#D2D2D7;font-size:1.3rem">⚔️</div>',unsafe_allow_html=True)
            with tc3:
                st.markdown(f'<div class="t-block" style="flex-direction:row-reverse;text-align:right"><span class="t-flag">{flag(away)}</span><div><div class="t-name">{away}</div><div class="t-sub" style="text-align:right">ELO {a_elo} · FIFA #{a_rank}</div></div></div>',unsafe_allow_html=True)

            st.markdown(f"""
            <div class="p-bar-wrap">
              <div class="p-bar-labels">
                <span class="p-bar-lbl" style="color:#1D1D1F">{home}</span>
                <span class="p-bar-lbl" style="color:#86868B">{away}</span>
              </div>
              <div class="p-bar"><div class="p-seg-h" style="width:{hpct:.1f}%"></div><div class="p-seg-a" style="width:{apct:.1f}%"></div></div>
              <div class="p-nums">
                <div><div class="p-pct-h">%{hpct:.1f}</div><div class="p-note">ev sahibi galibiyeti</div></div>
                <div style="text-align:right"><div class="p-pct-a">%{apct:.1f}</div><div class="p-note">deplasman / beraberlik</div></div>
              </div>
            </div>""",unsafe_allow_html=True)

            diff=abs(hpct-apct)
            if hpct>apct:
                conf="Yüksek güvenle" if diff>20 else "Hafif favori olarak"
                st.markdown(f'<div class="verdict v-h">🔵&nbsp;&nbsp;{conf} <strong>{home}</strong> kazanır · Olasılık %{hpct:.1f}</div>',unsafe_allow_html=True)
            elif apct>hpct:
                conf="Yüksek güvenle" if diff>20 else "Hafif favori olarak"
                st.markdown(f'<div class="verdict v-a">🔴&nbsp;&nbsp;{conf} <strong>{away}</strong> avantajlı · Olasılık %{apct:.1f}</div>',unsafe_allow_html=True)
            else:
                st.markdown('<div class="verdict v-e">⚪&nbsp;&nbsp;Model iki takımı <strong>eşit</strong> görüyor</div>',unsafe_allow_html=True)

            st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
            hl,hr_ = st.columns(2)
            with hl:
                st.markdown('<div class="sec-lbl">Son Karşılaşmalar (H2H)</div>',unsafe_allow_html=True)
                h2h = get_h2h(home,away,5)
                if h2h.empty:
                    st.markdown('<p style="color:#86868B;font-size:.85rem; text-align:center;">Karşılaşma verisi bulunamadı.</p>',unsafe_allow_html=True)
                else:
                    html_h2h=""
                    for _,r in h2h.iterrows():
                        hs_=int(r["home_score"]); aws_=int(r["away_score"]); yr=str(r["date"])[:4]
                        ht_d=home if resolve_res(home)==r["home_team"] else away
                        at_d=away if ht_d==home else home
                        html_h2h+=f'<div class="h2h-row"><span class="h2h-th">{flag(ht_d)} {ht_d}</span><span class="h2h-yr">{yr}</span><span class="h2h-sc">{hs_}–{aws_}</span><span class="h2h-ta">{flag(at_d)} {at_d}</span></div>'
                    st.markdown(f'<div class="a-card" style="padding:1rem 1.25rem">{html_h2h}</div>',unsafe_allow_html=True)
            with hr_:
                st.markdown('<div class="sec-lbl">Son 5 Maç Formu</div>',unsafe_allow_html=True)
                for t in [home,away]:
                    form=get_form(t,5)
                    dots="".join([f'<span class="fd fd-{r}">{r}</span>' for r in form]) if form else "–"
                    st.markdown(f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:.75rem;background:#FFFFFF;border-radius:12px;padding:.75rem 1.2rem;border:1px solid #E5E5EA"><div style="display:flex;align-items:center;gap:.75rem"><span style="font-size:1.4rem">{flag(t)}</span><span style="font-size:.9rem;font-weight:600;color:#1D1D1F;">{t}</span></div><div>{dots}</div></div>',unsafe_allow_html=True)

            st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
            ha_,ht_,hfi_ = wc_stats(home); aa_,at_,afi_ = wc_stats(away)
            hf_,hsc_,hcn_ = form_stats(home); af_,asc_,acn_ = form_stats(away)

            def sw(hv,av,higher=True):
                try:
                    hw=(higher and float(hv)>float(av)) or (not higher and float(hv)<float(av))
                    aw=(higher and float(av)>float(hv)) or (not higher and float(av)<float(hv))
                    return "sv"+(" sv-w" if hw else ""),"sv sv-a"+(" sv-w" if aw else "")
                except: return "sv","sv sv-a"

            def srow(lbl,hv,av,fmt="{}",higher=True):
                hc,ac=sw(hv,av,higher)
                try: hd=fmt.format(float(hv))
                except: hd=str(hv)
                try: ad=fmt.format(float(av))
                except: ad=str(av)
                return f'<div class="stat-row"><div class="{hc}">{hd}</div><div class="sk">{lbl}</div><div class="{ac}">{ad}</div></div>'

            st.markdown('<div class="sec-lbl">Takım Karşılaştırması</div>',unsafe_allow_html=True)
            rows_html=(srow("ELO Puanı",h_elo,a_elo,"{:.0f}")+srow("FIFA Sıralaması",h_rank,a_rank,"#{:.0f}",higher=False)+
                srow("Son Form",hf_,af_,"{:.2f}")+srow("Ort. Atılan Gol",hsc_,asc_,"{:.2f}")+
                srow("Ort. Yenilen Gol",hcn_,acn_,"{:.2f}",higher=False)+srow("DK Katılım",ha_,aa_,"{:.0f}")+
                srow("Şampiyonluk",ht_,at_,"{:.0f}")+srow("Final",hfi_,afi_,"{:.0f}"))
            st.markdown(f'<div class="a-card" style="padding:1rem 1.5rem">{rows_html}</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — TURNUVA SİMÜLATÖRÜ (FIFA Dünya Kupası 2026 Formatı)
# ══════════════════════════════════════════════════════════════════════════════
with tab_sim:

    # ── Mod seçimi ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-lbl">Simülasyon Modu</div>',unsafe_allow_html=True)

    # 1. LOKAL CSS: Sadece bu bileşeni ezip kendi içinde merkeze zorlar
    st.markdown("""
    <style>
    div[data-testid="stRadio"] { display: flex !important; align-items: center !important; justify-content: center !important; width: 100% !important; margin-bottom: 0.5rem !important; }
    div[data-testid="stRadio"] > div[role="radiogroup"] { display: flex !important; justify-content: center !important; gap: 3rem !important; }
    </style>
    """, unsafe_allow_html=True)

    # 2. KOLON HAPSİ: Bileşeni sola yapışmasın diye simetrik kolonların içine sıkıştırır
    _, center_col, _ = st.columns([2, 4, 1])
    with center_col:
        sel = st.radio("mod", ["🤖  Yapay Zeka Simülasyonu", "✏️  Kendi Tahminlerimi Oluştur"],
            key="sim_mode_radio", horizontal=True, label_visibility="collapsed")
        
    sim_mode = "AI" if sel.startswith("🤖") else "User"
    if sim_mode != st.session_state.sim_mode:
        reset_tour(); st.session_state.sim_mode=sim_mode; st.rerun()

    # ── Grup bilgi başlığı ─────────────────────────────────────────────────────
    grp_keys = list(WC_GROUPS.keys())  # A..L

    def render_plain_grp(g,teams):
        rows="".join(f'<div class="grp-row"><span class="grp-flag">{flag(t)}</span><span class="grp-name">{t}</span></div>' for t in teams)
        return f'<div class="grp-card"><div class="grp-hdr">GRUP {g}</div>{rows}</div>'

    def render_std_grp(g,sorted_t,stats_d,best3=None):
        if best3 is None: best3=[]
        col_h='<div class="grp-col-hdr"><span style="width:14px"></span><span style="width:1.5rem"></span><span style="flex:1">Takım</span><span style="width:18px;text-align:center">O</span><span style="width:18px;text-align:center">G</span><span style="width:18px;text-align:center">M</span><span style="width:22px;text-align:right">P</span></div>'
        rows=""
        for i,t in enumerate(sorted_t):
            s=stats_d.get(t,{"P":0,"W":0,"L":0,"Pts":0})
            qc="q-g" if i<2 else("q-o" if t in best3 else "")
            rows+=f'<div class="grp-row {qc}"><span class="grp-pos">{i+1}</span><span class="grp-flag">{flag(t)}</span><span class="grp-name">{t}</span><span class="grp-stat">{s["P"]}</span><span class="grp-stat">{s["W"]}</span><span class="grp-stat">{s["L"]}</span><span class="grp-pts">{s["Pts"]}</span></div>'
        return f'<div class="grp-card"><div class="grp-hdr">GRUP {g}</div>{col_h}{rows}</div>'

    # ══════════════════════════════════════════════════════════════════════════
    # GRUP AŞAMASI
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="sec-lbl">Grup Aşaması — 12 Grup · 48 Takım</div>',unsafe_allow_html=True)

    # ── AI MODE ────────────────────────────────────────────────────────────────
    if sim_mode=="AI":
        if not st.session_state.group_stage_done:
            for ri in range(0,12,4):
                cols=st.columns(4)
                for ci,g in enumerate(grp_keys[ri:ri+4]):
                    with cols[ci]: st.markdown(render_plain_grp(g,WC_GROUPS[g]),unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
            _,sc,_ = st.columns([2,3,2])
            with sc:
                st.markdown('<span class="btn-hint">12 grubu yapay zeka ile otomatik analiz et</span>',unsafe_allow_html=True)
                if st.button("🤖  Grup Aşamasını Simüle Et", key="btn_ai_sim", type="primary", use_container_width=True):
                    with st.spinner("Yapay zeka turnuvayı analiz ediyor..."):
                        time.sleep(0.6)
                        for g,teams in WC_GROUPS.items():
                            st_,stats_,matches_ = simulate_group(teams)
                            st.session_state.group_standings[g]=st_
                            st.session_state.group_stats[g]=stats_
                            st.session_state.group_matches[g]=matches_
                        b3=get_best_thirds(st.session_state.group_standings,st.session_state.group_stats,8)
                        st.session_state.r32_bracket=build_r32(st.session_state.group_standings,b3)
                        st.session_state.group_stage_done=True
                    st.rerun()
        else:
            b3=get_best_thirds(st.session_state.group_standings,st.session_state.group_stats,8)
            st.markdown('<div class="b-ok">✅ Grup aşaması tamamlandı. İlk 2\'şer takım (24) + en iyi 8 üçüncü = 32 takım Son 32 turuna geçti.</div>',unsafe_allow_html=True)
            for ri in range(0,12,4):
                cols=st.columns(4)
                for ci,g in enumerate(grp_keys[ri:ri+4]):
                    with cols[ci]:
                        st.markdown(render_std_grp(g,st.session_state.group_standings.get(g,[]),
                            st.session_state.group_stats.get(g,{}),b3),unsafe_allow_html=True)
                st.markdown("<br>",unsafe_allow_html=True)
            st.markdown('<div class="b-info">⭐ <strong>En İyi 8 Üçüncü:</strong> '+" · ".join(f"{flag(t)} {t}" for t in b3)+'</div>',unsafe_allow_html=True)

    # ── USER MODE ──────────────────────────────────────────────────────────────
    else:
        locked = st.session_state.group_stage_done
        total_picked=0

        for g,teams_g in WC_GROUPS.items():
            for h_t,a_t in combinations(teams_g,2):
                if st.session_state.get(f"up_{g}_{h_t}_vs_{a_t}"): total_picked+=1

        pct_done = total_picked/TOTAL_GROUP_MATCHES
        st.markdown(f'<p style="font-size:.85rem;color:#86868B;margin:.25rem 0 .5rem;text-align:center;font-weight:600;">Seçilen Maçlar: {total_picked} / {TOTAL_GROUP_MATCHES}</p>',unsafe_allow_html=True)
        _,pb,_ = st.columns([1,6,1])
        with pb:
            st.markdown(f'<div class="prog-bar"><div class="prog-fill" style="width:{pct_done*100:.0f}%"></div></div>',unsafe_allow_html=True)

                # ── Grup puan durumu kartları (4 kolon) ──────────────────────────────
        for ri in range(0,12,4):
            cols=st.columns(4)
            for ci,g in enumerate(grp_keys[ri:ri+4]):
                teams_g=WC_GROUPS[g]
                matches_g=list(combinations(teams_g,2))
                live={t:{"P":0,"W":0,"L":0,"Pts":0} for t in teams_g}
                picks_done=0
                for h_t,a_t in matches_g:
                    pk=st.session_state.get(f"up_{g}_{h_t}_vs_{a_t}")
                    if pk:
                        picks_done+=1; lo=a_t if pk==h_t else h_t
                        live[pk]["W"]+=1; live[pk]["Pts"]+=3; live[lo]["L"]+=1
                        live[h_t]["P"]+=1; live[a_t]["P"]+=1
                sl=sorted(teams_g,key=lambda t:(-live[t]["Pts"],-live[t]["W"],-wc_stats(t)[0]))
                with cols[ci]:
                    rows2="".join(
                        f'<div class="grp-row {"q-g" if i<2 else ""}"><span class="grp-pos">{i+1}</span><span class="grp-flag">{flag(t)}</span><span class="grp-name">{t}</span><span class="grp-stat">{live[t]["P"]}</span><span class="grp-pts">{live[t]["Pts"]}</span></div>'
                        for i,t in enumerate(sl))
                    col_h2='<div class="grp-col-hdr"><span style="width:14px"></span><span style="width:1.5rem"></span><span style="flex:1">Takım</span><span style="width:18px;text-align:center">O</span><span style="width:22px;text-align:right">P</span></div>'
                    st.markdown(f'<div class="grp-card"><div class="grp-hdr">GRUP {g} &nbsp;<span style="font-weight:500;color:#86868B">({picks_done}/{len(matches_g)})</span></div>{col_h2}{rows2}</div>',unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)

        # ── Maç seçim paneli (geniş layout, kolonların dışında) ─────────────
        if not locked:
            st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
            st.markdown('<div class="sec-lbl">Grup Maçlarını Seç — Kazananı Belirleyin</div>',unsafe_allow_html=True)

            for ri in range(0,12,3):
                pick_cols = st.columns(3)
                for ci,g in enumerate(grp_keys[ri:ri+3]):
                    
                    matches_g = list(combinations(teams_g,2))
                    n_picked = len([1 for h,a in matches_g if st.session_state.get(f"up_{g}_{h}_vs_{a}")])
                    with pick_cols[ci]:
                        st.markdown(
                            f'<div style="background:#F5F5F7;border:1px solid #E5E5EA;'
                            f'border-radius:12px 12px 0 0;padding:.5rem .85rem;font-size:.7rem;'
                            f'font-weight:800;letter-spacing:2px;text-transform:uppercase;'
                            f'text-align:center;color:#1D1D1F;">'
                            f'GRUP {g} &nbsp;<span style="font-weight:500;color:#86868B;">'
                            f'({n_picked}/{len(matches_g)})</span></div>',
                            unsafe_allow_html=True)
                        with st.container():
                            for h_t,a_t in matches_g:
                                key = f"up_{g}_{h_t}_vs_{a_t}"
                                current = st.session_state.get(key)
                                idx = None
                                if current == h_t: idx = 0
                                elif current == a_t: idx = 1
                                st.radio(
                                    f"{flag(h_t)} {h_t}  —  {flag(a_t)} {a_t}",
                                    options=[h_t, a_t],
                                    key=key,
                                    horizontal=True,
                                    index=idx,
                                    format_func=lambda x, h=h_t, a=a_t: f"{flag(x)} {x}",
                                )
                st.markdown("<br>",unsafe_allow_html=True)              


        if not locked:
            all_done = (total_picked==TOTAL_GROUP_MATCHES)
            if not all_done:
                st.markdown(f'<div class="b-warn">⚠️ {TOTAL_GROUP_MATCHES-total_picked} maç tahmini eksik. Tümünü tamamlayın veya Yapay Zeka moduna geçin.</div>',unsafe_allow_html=True)
            _,sc2,_ = st.columns([2,3,2])
            with sc2:
                st.markdown('<span class="btn-hint">72 maçı tamamladıktan sonra eleme turunu kilitle</span>',unsafe_allow_html=True)
                if st.button("📊  Ağacı Oluştur", key="btn_user_build", disabled=not all_done, type="primary", use_container_width=True):
                    for g,teams_g in WC_GROUPS.items():
                        picks_d={f"{h}_vs_{a}":st.session_state.get(f"up_{g}_{h}_vs_{a}")
                                 for h,a in combinations(teams_g,2) if st.session_state.get(f"up_{g}_{h}_vs_{a}")}
                        st_,stats_,matches_ = simulate_group_from_picks(teams_g,picks_d)
                        st.session_state.group_standings[g]=st_
                        st.session_state.group_stats[g]=stats_
                        st.session_state.group_matches[g]=matches_
                    b3=get_best_thirds(st.session_state.group_standings,st.session_state.group_stats,8)
                    st.session_state.r32_bracket=build_r32(st.session_state.group_standings,b3)
                    st.session_state.group_stage_done=True
                    st.rerun()
        else:
            b3=get_best_thirds(st.session_state.group_standings,st.session_state.group_stats,8)
            st.markdown('<div class="b-ok">✅ Grup aşaması kilitlendi!</div>',unsafe_allow_html=True)
            st.markdown('<div class="b-info">⭐ <strong>En İyi 8 Üçüncü:</strong> '+" · ".join(f"{flag(t)} {t}" for t in b3)+'</div>',unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # ELEME AŞAMASI
    # ══════════════════════════════════════════════════════════════════════════
    if st.session_state.group_stage_done:
        r32 = st.session_state.r32_bracket
        ko  = st.session_state.ko_results

        def kw(k): return ko.get(k)

        def bkt_html(t1,t2,rkey):
            w=kw(rkey)
            def row(t):
                if not t or t=="TBD": return '<div class="bkt-team"><span class="bkt-tbd">TBD</span></div>'
                c=" bkt-win" if w==t else ""
                return f'<div class="bkt-team{c}"><span class="bkt-fl">{flag(t)}</span><span class="bkt-nm">{t}</span></div>'
            return f'<div class="bkt-match">{row(t1)}{row(t2)}</div>'

        KO_ORDER=([f"r32_{i}" for i in range(16)]+[f"r16_{i}" for i in range(8)]+
                  [f"qf_{i}" for i in range(4)]+[f"sf_{i}" for i in range(2)]+["final"])

        def clear_down(changed):
            try:
                idx=KO_ORDER.index(changed)
                for k in KO_ORDER[idx+1:]: ko.pop(k,None)
                st.session_state.champion=None
            except ValueError: pass

        def pick_btns(t1,t2,rkey,col_ctx):
            if not t1 or not t2 or "TBD" in(t1,t2) or kw(rkey): return
            with col_ctx:
                st.markdown('<div style="text-align:center; font-size:0.7rem; color:#86868B; margin-bottom:6px; font-weight:600;">Kazananı Seç:</div>', unsafe_allow_html=True)
                if st.button(f"{flag(t1)} {t1}", key=f"kb_{rkey}_1", use_container_width=True):
                    ko[rkey]=t1; clear_down(rkey); st.rerun()
                if st.button(f"{flag(t2)} {t2}", key=f"kb_{rkey}_2", use_container_width=True):
                    ko[rkey]=t2; clear_down(rkey); st.rerun()
                st.markdown('<div style="height:1.5rem;"></div>', unsafe_allow_html=True)

        st.markdown('<div class="hr"></div>',unsafe_allow_html=True)

        # ── SON 32 ────────────────────────────────────────────────────────────
        st.markdown('<div class="bkt-rnd">SON 32 (Round of 32)</div>',unsafe_allow_html=True)

        col_groups = [(0,1,2,3),(4,5,6,7),(8,9,10,11),(12,13,14,15)]
        r32_cols = st.columns(4)
        for ci,mg in enumerate(col_groups):
            with r32_cols[ci]:
                for m in mg:
                    if m<len(r32):
                        t1,t2=r32[m]
                        st.markdown(bkt_html(t1,t2,f"r32_{m}"),unsafe_allow_html=True)

        r32_done = all(kw(f"r32_{i}") for i in range(min(16,len(r32))))

        if sim_mode=="AI" and not r32_done:
            _,sc3,_=st.columns([2,3,2])
            with sc3:
                st.markdown('<span class="btn-hint">16 eleme maçını yapay zeka ile belirle</span>',unsafe_allow_html=True)
                if st.button("▶️  Son 32 Turunu Simüle Et", key="btn_r32", type="primary", use_container_width=True):
                    with st.spinner("Son 32 simüle ediliyor..."):
                        time.sleep(0.4)
                        for i,(t1,t2) in enumerate(r32):
                            if not kw(f"r32_{i}") and t1 and t2 and "TBD" not in(t1,t2):
                                hp_,ap_=predict_match(t1,t2)
                                ko[f"r32_{i}"]=t1 if hp_>ap_ else t2
                    st.rerun()
        elif sim_mode=="User" and not r32_done:
            st.markdown('<p style="color:#86868B;font-size:.85rem;font-weight:600;text-align:center;margin:1rem 0">Son 32 kazananlarını seçin:</p>',unsafe_allow_html=True)
            pick_cols=st.columns(4)
            for ci,mg in enumerate(col_groups):
                for m in mg:
                    if m<len(r32):
                        t1,t2=r32[m]
                        pick_btns(t1,t2,f"r32_{m}",pick_cols[ci])

        # ── SON 16 ────────────────────────────────────────────────────────────
        if r32_done:
            r16=[
                (kw("r32_0"), kw("r32_1")), (kw("r32_2"), kw("r32_3")),
                (kw("r32_4"), kw("r32_5")), (kw("r32_6"), kw("r32_7")),
                (kw("r32_8"), kw("r32_9")), (kw("r32_10"),kw("r32_11")),
                (kw("r32_12"),kw("r32_13")),(kw("r32_14"),kw("r32_15")),
            ]
            st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
            st.markdown('<div class="bkt-rnd">SON 16 (Round of 16)</div>',unsafe_allow_html=True)
            r16_cols=st.columns(4)
            r16_col_groups=[(0,1),(2,3),(4,5),(6,7)]
            for ci,(ma,mb) in enumerate(r16_col_groups):
                with r16_cols[ci]:
                    for m in [ma,mb]:
                        t1,t2=r16[m]
                        st.markdown(bkt_html(t1 or "TBD",t2 or "TBD",f"r16_{m}"),unsafe_allow_html=True)

            r16_done=all(kw(f"r16_{i}") for i in range(8))
            if sim_mode=="AI" and not r16_done:
                _,sc4,_=st.columns([2,3,2])
                with sc4:
                    st.markdown('<span class="btn-hint">8 son 16 maçını yapay zeka ile belirle</span>',unsafe_allow_html=True)
                    if st.button("▶️  Son 16 Turunu Simüle Et", key="btn_r16", type="primary", use_container_width=True):
                        with st.spinner("Son 16 simüle ediliyor..."):
                            time.sleep(0.3)
                            for i,(t1,t2) in enumerate(r16):
                                if t1 and t2 and not kw(f"r16_{i}"):
                                    hp_,ap_=predict_match(t1,t2)
                                    ko[f"r16_{i}"]=t1 if hp_>ap_ else t2
                        st.rerun()
            elif sim_mode=="User" and not r16_done:
                st.markdown('<p style="color:#86868B;font-size:.85rem;font-weight:600;text-align:center;margin:1rem 0">Son 16 kazananlarını seçin:</p>',unsafe_allow_html=True)
                for ci,(ma,mb) in enumerate(r16_col_groups):
                    for m in [ma,mb]:
                        t1,t2=r16[m]
                        pick_btns(t1 or "TBD",t2 or "TBD",f"r16_{m}",r16_cols[ci])

            # ── ÇEYREK FİNAL ─────────────────────────────────────────────────
            if r16_done:
                qf=[(kw("r16_0"),kw("r16_1")),(kw("r16_2"),kw("r16_3")),
                    (kw("r16_4"),kw("r16_5")),(kw("r16_6"),kw("r16_7"))]
                st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
                st.markdown('<div class="bkt-rnd">ÇEYREK FİNAL (Quarter Finals)</div>',unsafe_allow_html=True)
                qf_cols=st.columns(4)
                for i,(t1,t2) in enumerate(qf):
                    with qf_cols[i]:
                        st.markdown(bkt_html(t1 or "TBD",t2 or "TBD",f"qf_{i}"),unsafe_allow_html=True)

                qf_done=all(kw(f"qf_{i}") for i in range(4))
                if sim_mode=="AI" and not qf_done:
                    _,sc5,_=st.columns([2,3,2])
                    with sc5:
                        st.markdown('<span class="btn-hint">4 çeyrek final maçını yapay zeka ile belirle</span>',unsafe_allow_html=True)
                        if st.button("▶️  Çeyrek Finalleri Simüle Et", key="btn_qf", type="primary", use_container_width=True):
                            with st.spinner("Çeyrek finaller simüle ediliyor..."):
                                time.sleep(0.3)
                                for i,(t1,t2) in enumerate(qf):
                                    if t1 and t2:
                                        hp_,ap_=predict_match(t1,t2)
                                        ko[f"qf_{i}"]=t1 if hp_>ap_ else t2
                            st.rerun()
                elif sim_mode=="User" and not qf_done:
                    st.markdown('<p style="color:#86868B;font-size:.85rem;font-weight:600;text-align:center;margin:1rem 0">Çeyrek final kazananlarını seçin:</p>',unsafe_allow_html=True)
                    for i,(t1,t2) in enumerate(qf):
                        pick_btns(t1 or "TBD",t2 or "TBD",f"qf_{i}",qf_cols[i])

                # ── YARI FİNAL ───────────────────────────────────────────────
                if qf_done:
                    sf=[(kw("qf_0"),kw("qf_1")),(kw("qf_2"),kw("qf_3"))]
                    st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
                    st.markdown('<div class="bkt-rnd">YARI FİNAL (Semi Finals)</div>',unsafe_allow_html=True)
                    _,sf1c,_,sf2c,_=st.columns([1,3,1,3,1])
                    sf_ctxs=[sf1c,sf2c]
                    for i,(t1,t2) in enumerate(sf):
                        with sf_ctxs[i]:
                            st.markdown(bkt_html(t1 or "TBD",t2 or "TBD",f"sf_{i}"),unsafe_allow_html=True)

                    sf_done=all(kw(f"sf_{i}") for i in range(2))
                    if sim_mode=="AI" and not sf_done:
                        _,sc6,_=st.columns([2,3,2])
                        with sc6:
                            st.markdown('<span class="btn-hint">2 yarı final maçını yapay zeka ile belirle</span>',unsafe_allow_html=True)
                            if st.button("▶️  Yarı Finalleri Simüle Et", key="btn_sf", type="primary", use_container_width=True):
                                with st.spinner("Yarı finaller simüle ediliyor..."):
                                    time.sleep(0.3)
                                    for i,(t1,t2) in enumerate(sf):
                                        if t1 and t2:
                                            hp_,ap_=predict_match(t1,t2)
                                            ko[f"sf_{i}"]=t1 if hp_>ap_ else t2
                                st.rerun()
                    elif sim_mode=="User" and not sf_done:
                        st.markdown('<p style="color:#86868B;font-size:.85rem;font-weight:600;text-align:center;margin:1rem 0">Yarı final kazananlarını seçin:</p>',unsafe_allow_html=True)
                        for i,(t1,t2) in enumerate(sf):
                            pick_btns(t1 or "TBD",t2 or "TBD",f"sf_{i}",sf_ctxs[i])

                    # ── FİNAL ────────────────────────────────────────────────
                    if sf_done:
                        ft1,ft2=kw("sf_0"),kw("sf_1")
                        st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
                        st.markdown('<div class="bkt-rnd">FİNAL</div>',unsafe_allow_html=True)
                        _,fc,_=st.columns([2,3,2])
                        with fc:
                            st.markdown(bkt_html(ft1 or "TBD",ft2 or "TBD","final"),unsafe_allow_html=True)

                        final_done=bool(kw("final"))
                        if sim_mode=="AI" and not final_done:
                            _,sc7,_=st.columns([2,3,2])
                            with sc7:
                                st.markdown('<span class="btn-hint">Dünya Kupası şampiyonunu yapay zeka ile belirle</span>',unsafe_allow_html=True)
                                if st.button("▶️  Finali Simüle Et", key="btn_final", type="primary", use_container_width=True):
                                    with st.spinner("Final simüle ediliyor..."):
                                        time.sleep(0.4)
                                        if ft1 and ft2:
                                            hp_,ap_=predict_match(ft1,ft2)
                                            champ=ft1 if hp_>ap_ else ft2
                                            ko["final"]=champ; st.session_state.champion=champ
                                    st.rerun()
                        elif sim_mode=="User" and not final_done:
                            st.markdown('<p style="color:#86868B;font-size:.85rem;font-weight:600;text-align:center;margin:1rem 0">Şampiyonu seçin:</p>',unsafe_allow_html=True)
                            pick_btns(ft1 or "TBD",ft2 or "TBD","final",fc)

                        # ── ŞAMPİYON ─────────────────────────────────────────
                        champ=kw("final")
                        if champ:
                            st.session_state.champion=champ
                            st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
                            _,cc,_=st.columns([1,3,1])
                            with cc:
                                st.markdown(f"""
                                <div class="champ-wrap">
                                  <div class="champ-trophy">🏆</div>
                                  <div class="champ-lbl">FIFA Dünya Kupası 2026 Şampiyonu</div>
                                  <div class="champ-fl">{flag(champ)}</div>
                                  <div class="champ-nm">{champ}</div>
                                </div>""",unsafe_allow_html=True)

                            if sim_mode=="User":
                                st.markdown('<div class="hr"></div>',unsafe_allow_html=True)
                                st.markdown('<div class="sec-lbl">Tahmin Özetiniz</div>',unsafe_allow_html=True)
                                sum_cols=st.columns(5)
                                rounds=[("Son 32",[f"r32_{i}" for i in range(16)]),
                                        ("Son 16",[f"r16_{i}" for i in range(8)]),
                                        ("Çeyrek F.",[f"qf_{i}" for i in range(4)]),
                                        ("Yarı F.",[f"sf_{i}" for i in range(2)]),
                                        ("Final",["final"])]
                                for ci,(lbl,rkeys) in enumerate(rounds):
                                    with sum_cols[ci]:
                                        winners=[kw(k) for k in rkeys if kw(k)]
                                        rows_s="".join(f'<div class="grp-row q-g"><span class="grp-flag">{flag(w)}</span><span class="grp-name">{w}</span></div>' for w in winners)
                                        st.markdown(f'<div class="grp-card"><div class="grp-hdr">{lbl}</div>{rows_s}</div>',unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# YASAL UYARI (Her iki sekmenin altında görünür)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="disclaimer">
  <strong>Yasal Uyarı / Disclaimer:</strong> Bu uygulama tamamen açık kaynaklı veri setleri ve makine öğrenmesi
  algoritmaları kullanılarak eğlence ve portfolyo amacıyla geliştirilmiş bir simülasyondur. Burada üretilen
  tahminler ve olasılıklar kesinlikle yatırım, bahis veya iddaa tavsiyesi niteliği taşımamaktadır.
  Kullanıcıların bu sonuçlara dayanarak alacağı finansal kararlardan ve doğabilecek maddi/manevi
  zararlardan geliştirici hiçbir şekilde sorumlu tutulamaz.
</div>""",unsafe_allow_html=True)