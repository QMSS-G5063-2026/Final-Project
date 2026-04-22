"""
NYC Job Market Explorer — QMSS G5063 Final Project
Streamlit multi-page interactive dashboard
"""

import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import nltk
from nltk.corpus import stopwords
import io

# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Job Market Explorer",
    page_icon="🗽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── colour palette ─────────────────────────────────────────────────────────────
PALETTE = {
    "navy":    "#0A2342",
    "blue":    "#1D4E89",
    "sky":     "#4CA3DD",
    "teal":    "#2E8B8B",
    "gold":    "#F7B731",
    "orange":  "#E8703A",
    "red":     "#C0392B",
    "light":   "#F5F7FA",
    "mid":     "#DCE3ED",
    "text":    "#1A1A2E",
}
BOROUGH_COLORS = {
    "Manhattan":   "#1D4E89",
    "Brooklyn":    "#2E8B8B",
    "Queens":      "#F7B731",
    "Bronx":       "#E8703A",
    "Staten Island":"#C0392B",
    "Unknown":     "#CCCCCC",
}

# ── global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* sidebar */
  section[data-testid="stSidebar"] {
      background: #0A2342;
  }
  section[data-testid="stSidebar"] * { color: #F5F7FA !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stMultiSelect label { color: #4CA3DD !important; font-weight: 600; }

  /* metric cards */
  div[data-testid="metric-container"] {
      background: #ffffff;
      border: 1px solid #DCE3ED;
      border-radius: 10px;
      padding: 14px 18px;
      box-shadow: 0 2px 8px rgba(10,35,66,0.07);
  }
  div[data-testid="metric-container"] label { color: #1D4E89 !important; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.05em; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0A2342 !important; font-size: 1.8rem; font-weight: 700; }

  /* page title */
  .page-header {
      background: linear-gradient(135deg, #0A2342 0%, #1D4E89 60%, #4CA3DD 100%);
      border-radius: 14px;
      padding: 28px 36px;
      margin-bottom: 28px;
      color: white;
  }
  .page-header h1 { margin:0; font-size: 2rem; font-weight: 700; }
  .page-header p  { margin:6px 0 0; opacity: 0.85; font-size: 1rem; }

  /* section headers */
  .section-header {
      color: #0A2342;
      font-size: 1.15rem;
      font-weight: 700;
      border-left: 4px solid #4CA3DD;
      padding-left: 10px;
      margin: 24px 0 12px;
  }

  /* plotly charts */
  .stPlotlyChart { border-radius: 10px; overflow: hidden; }

  /* tab style */
  button[data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ── data loading & preprocessing ───────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("NYC_Jobs.csv")

    # ── borough extraction ──────────────────────────────────────────────────────
    addr_lookup = {
        "96-05 HORACE HARDING": "Queens",
        "42-09 28": "Queens",
        "59-17 JUNCTION": "Queens",
        "255 GREENWICH": "Manhattan",
        "30-30 47": "Queens",
        "75-20 ASTORIA": "Queens",
        "59 MAIDEN LANE": "Manhattan",
        "22 READE": "Manhattan",
        "180 MAIDEN": "Manhattan",
        "3030 THIRD AVE": "Bronx",
        "421 EAST 26": "Manhattan",
        "777 THIRD": "Manhattan",
        "HAZEN ST": "Queens",
        "470 VANDERBILT": "Brooklyn",
        "CITY HALL": "Manhattan",
        "59 MAIDEN": "Manhattan",
        "30-30 THOMSON": "Queens",
        "43-06 DITMARS": "Queens",
    }

    def extract_borough(loc):
        if pd.isna(loc):
            return "Unknown"
        u = loc.upper()
        if "BROOKLYN" in u or "BKLYN" in u:
            return "Brooklyn"
        if "BRONX" in u or "3030 THIRD AVE" in u:
            return "Bronx"
        if any(x in u for x in ["QUEENS", "L I CITY", "LIC NY", "FLUSHING",
                                  "JAMAICA", "CORONA", "ASTORIA", "WOODSIDE",
                                  "FOREST HILLS", "JACKSON HEIGHTS"]):
            return "Queens"
        if "STATEN ISLAND" in u or "RICHMOND" in u:
            return "Staten Island"
        if any(x in u for x in ["MANHATTAN", "NEW YORK", "N.Y.", " NYC",
                                  "WALL ST", "BROADWAY", "CENTRE ST", "CHURCH ST",
                                  "GOLD ST", "WORTH ST", "PEARL ST", "FULTON",
                                  "LIBERTY", "WORLD TRADE", "LAFAYETTE", "BEAVER",
                                  "HUDSON", "WATER ST", "WILLIAM ST", "JOHN ST",
                                  "NASSAU ST", "FRANKFORT", "MAIDEN LANE",
                                  "SPRUCE ST", "PINE ST", "CORTLANDT", "VESEY",
                                  "RECTOR", "EXCHANGE", "BROAD ST", "BOWLING GREEN",
                                  "BATTERY", "WHITEHALL", "READE", "GREENWICH",
                                  "CITY HALL", "CENTRE ST", "CHAMBERS", "PARK PL",
                                  "MURRAY ST", "WARREN ST", "BARCLAY", "ANN ST",
                                  "DEY ST", "FULTON ST", "CORTLANDT ST"]):
            return "Manhattan"
        # lookup table for known addresses
        for key, val in addr_lookup.items():
            if key in u:
                return val
        return "Unknown"

    df["Borough"] = df["Work Location"].apply(extract_borough)

    # ── salary cleanup ───────────────────────────────────────────────────────────
    annual = df["Salary Frequency"] == "Annual"
    df = df[annual].copy()
    df = df[(df["Salary Range From"] > 10_000) & (df["Salary Range To"] < 300_000)].copy()
    df["Salary Mid"] = (df["Salary Range From"] + df["Salary Range To"]) / 2

    # ── agency short names ───────────────────────────────────────────────────────
    agency_map = {
        "DEPT OF HEALTH/MENTAL HYGIENE":    "Health/Mental Hygiene",
        "DEPT OF ENVIRONMENT PROTECTION":   "Environment Protection",
        "DEPT OF DESIGN & CONSTRUCTION":    "Design & Construction",
        "HRA/DEPT OF SOCIAL SERVICES":      "Social Services",
        "ADMIN FOR CHILDREN'S SVCS":        "Children's Services",
        "OFFICE OF THE MAYOR":              "Mayor's Office",
        "HOUSING PRESERVATION & DVLPMNT":   "Housing Preservation",
        "NYC HOUSING AUTHORITY":            "Housing Authority",
        "OFFICE OF MANAGEMENT & BUDGET":    "Management & Budget",
        "BRONX DISTRICT ATTORNEY":          "Bronx DA",
        "DEPT OF CITYWIDE ADMIN SVCS":      "Admin Services",
        "DEPT OF TRANSPORTATION":           "Transportation",
        "DEPT OF CORRECTION":               "Correction",
        "DEPT OF SANITATION":               "Sanitation",
        "DEPARTMENT OF FINANCE":            "Finance",
        "DEPT OF BUILDINGS":                "Buildings",
        "NYC FIRE DEPARTMENT":              "Fire Department",
        "POLICE DEPARTMENT":                "Police",
    }
    df["Agency Short"] = df["Agency"].map(agency_map).fillna(
        df["Agency"].str.title().str[:28]
    )

    # ── posting date ─────────────────────────────────────────────────────────────
    # No post date column — use Post Until as a proxy
    df["Post Until"] = pd.to_datetime(df["Post Until"], errors="coerce", format="%d-%b-%Y")

    return df


@st.cache_data
def get_stopwords():
    try:
        nltk.download("stopwords", quiet=True)
        sw = set(stopwords.words("english"))
    except Exception:
        sw = set()
    extra = {
        "experience", "work", "include", "including", "new", "york", "city",
        "must", "required", "ability", "knowledge", "skills", "skill",
        "position", "duties", "will", "years", "equivalent", "degree",
        "related", "working", "job", "employee", "employees", "applicants",
        "nyc", "department", "bureau", "office", "strong", "excellent",
        "demonstrated", "ability", "preferred", "responsible", "responsibilities",
        "may", "also", "able", "us", "one", "two", "three", "four", "five",
        "agency", "minimum", "qualification", "qualifications", "good", "using",
        "use", "within", "across", "well", "provide", "support", "services",
        "service", "ensure", "make", "including", "managing", "management",
        "team", "staff", "high", "school", "diploma", "baccalaureate", "college",
        "accredited", "field", "another", "either", "area", "type"
    }
    return sw | extra


@st.cache_data
def run_topic_model(texts, n_topics=8):
    sw = get_stopwords()
    vec = TfidfVectorizer(
        max_features=3000,
        min_df=3,
        max_df=0.85,
        ngram_range=(1, 2),
        stop_words=list(sw),
    )
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=300)
    W = nmf.fit_transform(X)
    feature_names = vec.get_feature_names_out()
    topics = {}
    for i, comp in enumerate(nmf.components_):
        top_idx = comp.argsort()[-12:][::-1]
        topics[f"Topic {i+1}"] = [(feature_names[j], comp[j]) for j in top_idx]
    return W, topics


@st.cache_data
def tfidf_by_category(df, col="Agency Short", min_docs=15):
    sw = get_stopwords()
    groups = df.groupby(col)["Job Description"].apply(lambda x: " ".join(x.dropna())).reset_index()
    groups = groups[groups["Job Description"].str.len() > 100]
    vec = TfidfVectorizer(
        max_features=5000, min_df=2, stop_words=list(sw), ngram_range=(1, 2)
    )
    X = vec.fit_transform(groups["Job Description"])
    fn = vec.get_feature_names_out()
    result = {}
    for i, row in groups.iterrows():
        idx = X[groups.index.get_loc(i)].toarray()[0].argsort()[-10:][::-1]
        result[row[col]] = [(fn[j], X[groups.index.get_loc(i), j]) for j in idx]
    return result


# ── load ────────────────────────────────────────────────────────────────────────
df = load_data()

# ── sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗽 NYC Job Market\n#### Explorer")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "💰 Salary Analysis", "🗺️ Geographic Insights",
         "📝 Text & Skills Analysis", "🏢 Agency Deep-Dive"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Global Filters**")

    borough_opts = ["All"] + sorted([b for b in df["Borough"].unique() if b != "Unknown"])
    sel_borough = st.selectbox("Borough", borough_opts)

    posting_opts = ["All", "External", "Internal"]
    sel_posting = st.selectbox("Posting Type", posting_opts)

    salary_min, salary_max = int(df["Salary Range From"].min()), int(df["Salary Range To"].max())
    sel_salary = st.slider("Annual Salary Range ($)", salary_min, salary_max,
                           (salary_min, salary_max), step=5000,
                           format="$%d")

    st.markdown("---")
    st.markdown(
        "<small style='opacity:0.6'>Data: NYC Open Data · Updated Apr 2026<br>"
        "QMSS G5063 · Spring 2026</small>",
        unsafe_allow_html=True,
    )

# ── apply filters ───────────────────────────────────────────────────────────────
fdf = df.copy()
if sel_borough != "All":
    fdf = fdf[fdf["Borough"] == sel_borough]
if sel_posting != "All":
    fdf = fdf[fdf["Posting Type"] == sel_posting]
fdf = fdf[
    (fdf["Salary Range From"] >= sel_salary[0]) &
    (fdf["Salary Range To"] <= sel_salary[1])
]

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("""
    <div class="page-header">
      <h1>🗽 NYC Job Market Explorer</h1>
      <p>Exploring 2,800+ active NYC government job postings — salaries, skills, geography, and more.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Postings", f"{len(fdf):,}")
    c2.metric("Agencies", f"{fdf['Agency'].nunique():,}")
    c3.metric("Median Salary", f"${fdf['Salary Mid'].median():,.0f}")
    c4.metric("Avg Positions", f"{fdf['# Of Positions'].mean():.1f}")
    c5.metric("External Posts", f"{(fdf['Posting Type']=='External').sum():,}")

    st.markdown("---")
    col_l, col_r = st.columns([1.3, 1])

    # ── top agencies bar ─────────────────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-header">Job Postings by Agency (Top 15)</div>', unsafe_allow_html=True)
        top_ag = fdf["Agency Short"].value_counts().head(15).reset_index()
        top_ag.columns = ["Agency", "Postings"]
        fig = px.bar(
            top_ag.sort_values("Postings"),
            x="Postings", y="Agency",
            orientation="h",
            color="Postings",
            color_continuous_scale=["#4CA3DD", "#0A2342"],
            text="Postings",
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            coloraxis_showscale=False,
            margin=dict(l=0, r=30, t=10, b=10),
            xaxis_title="Number of Postings", yaxis_title="",
            height=460,
            font=dict(family="Inter"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── borough donut ────────────────────────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-header">Distribution by Borough</div>', unsafe_allow_html=True)
        bor = fdf[fdf["Borough"] != "Unknown"]["Borough"].value_counts().reset_index()
        bor.columns = ["Borough", "Count"]
        fig2 = px.pie(
            bor, values="Count", names="Borough",
            hole=0.55,
            color="Borough",
            color_discrete_map=BOROUGH_COLORS,
        )
        fig2.update_traces(
            textposition="outside",
            textinfo="label+percent",
            pull=[0.03]*len(bor),
        )
        fig2.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=10, b=10),
            paper_bgcolor="white",
            height=460,
            font=dict(family="Inter"),
            annotations=[dict(text=f"<b>{len(fdf):,}</b><br>jobs", x=0.5, y=0.5,
                              font_size=16, showarrow=False, font_color="#0A2342")]
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── salary distribution ──────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Salary Distribution by Posting Type</div>', unsafe_allow_html=True)
    fig3 = go.Figure()
    colors = {"External": "#1D4E89", "Internal": "#F7B731"}
    for ptype, color in colors.items():
        subset = fdf[fdf["Posting Type"] == ptype]["Salary Mid"]
        if len(subset):
            fig3.add_trace(go.Histogram(
                x=subset,
                name=ptype,
                marker_color=color,
                opacity=0.75,
                nbinsx=40,
                hovertemplate="Salary: $%{x:,.0f}<br>Count: %{y}<extra>%s</extra>" % ptype,
            ))
    fig3.update_layout(
        barmode="overlay",
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Midpoint Annual Salary ($)",
        yaxis_title="Number of Postings",
        legend_title="Posting Type",
        margin=dict(l=0, r=0, t=10, b=10),
        height=300,
        font=dict(family="Inter"),
    )
    fig3.update_xaxes(tickformat="$,.0f", showgrid=True, gridcolor="#DCE3ED")
    fig3.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
    st.plotly_chart(fig3, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SALARY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "💰 Salary Analysis":
    st.markdown("""
    <div class="page-header">
      <h1>💰 Salary Analysis</h1>
      <p>Compare salary distributions across agencies, boroughs, and posting types.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📊 By Agency", "🗺️ By Borough", "📈 Salary Ranges"])

    # ── tab 1: by agency ─────────────────────────────────────────────────────────
    with tab1:
        top_n = st.slider("Number of agencies to show", 5, 25, 15)
        top_agencies = fdf.groupby("Agency Short")["Salary Mid"].median().nlargest(top_n).index
        plot_df = fdf[fdf["Agency Short"].isin(top_agencies)].copy()
        order = plot_df.groupby("Agency Short")["Salary Mid"].median().sort_values().index.tolist()

        fig = go.Figure()
        for ag in order:
            sub = plot_df[plot_df["Agency Short"] == ag]["Salary Mid"]
            fig.add_trace(go.Box(
                x=sub,
                name=ag,
                orientation="h",
                marker_color="#1D4E89",
                line_color="#0A2342",
                fillcolor="rgba(76,163,221,0.3)",
                boxmean="sd",
                hovertemplate=f"<b>{ag}</b><br>Salary: $%{{x:,.0f}}<extra></extra>",
            ))
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Annual Salary ($)",
            xaxis_tickformat="$,.0f",
            showlegend=False,
            height=max(400, top_n * 30),
            margin=dict(l=160, r=20, t=20, b=40),
            font=dict(family="Inter"),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig, use_container_width=True)

        # salary table
        tbl = (fdf.groupby("Agency Short")["Salary Mid"]
               .agg(["median", "mean", "min", "max", "count"])
               .reset_index()
               .rename(columns={"Agency Short": "Agency", "median": "Median",
                                 "mean": "Mean", "min": "Min", "max": "Max",
                                 "count": "# Postings"})
               .sort_values("Median", ascending=False))
        for col in ["Median", "Mean", "Min", "Max"]:
            tbl[col] = tbl[col].apply(lambda x: f"${x:,.0f}")
        st.dataframe(tbl, use_container_width=True, hide_index=True)

    # ── tab 2: by borough ────────────────────────────────────────────────────────
    with tab2:
        bdf = fdf[fdf["Borough"] != "Unknown"].copy()

        col1, col2 = st.columns(2)
        with col1:
            bor_median = bdf.groupby("Borough")["Salary Mid"].median().reset_index()
            bor_median.columns = ["Borough", "Median Salary"]
            fig = px.bar(
                bor_median.sort_values("Median Salary", ascending=False),
                x="Borough", y="Median Salary",
                color="Borough",
                color_discrete_map=BOROUGH_COLORS,
                text="Median Salary",
                title="Median Salary by Borough",
            )
            fig.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
            fig.update_layout(
                showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                yaxis_tickformat="$,.0f", height=380,
                font=dict(family="Inter"), title_font_size=14,
                margin=dict(t=40, b=20),
            )
            fig.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.violin(
                bdf, x="Borough", y="Salary Mid",
                color="Borough",
                color_discrete_map=BOROUGH_COLORS,
                box=True,
                title="Salary Distribution by Borough",
            )
            fig2.update_layout(
                showlegend=False, plot_bgcolor="white", paper_bgcolor="white",
                yaxis_tickformat="$,.0f", height=380,
                font=dict(family="Inter"), title_font_size=14,
                margin=dict(t=40, b=20),
            )
            fig2.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
            st.plotly_chart(fig2, use_container_width=True)

        # scatter: salary from vs to
        st.markdown('<div class="section-header">Salary Range Width vs. Midpoint</div>', unsafe_allow_html=True)
        scatter_df = bdf.copy()
        scatter_df["Salary Range Width"] = scatter_df["Salary Range To"] - scatter_df["Salary Range From"]
        fig3 = px.scatter(
            scatter_df.sample(min(len(scatter_df), 600), random_state=42),
            x="Salary Mid", y="Salary Range Width",
            color="Borough",
            color_discrete_map=BOROUGH_COLORS,
            hover_data=["Agency Short", "Business Title"],
            opacity=0.7,
            size_max=8,
        )
        fig3.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Midpoint Salary ($)", yaxis_title="Salary Range Width ($)",
            xaxis_tickformat="$,.0f", yaxis_tickformat="$,.0f",
            height=360,
            font=dict(family="Inter"),
            margin=dict(t=10, b=20),
        )
        fig3.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
        fig3.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig3, use_container_width=True)

    # ── tab 3: salary ranges ─────────────────────────────────────────────────────
    with tab3:
        st.markdown('<div class="section-header">Salary Range Explorer — Top Agencies</div>', unsafe_allow_html=True)

        top_ag_range = st.multiselect(
            "Select agencies to compare",
            options=sorted(fdf["Agency Short"].unique()),
            default=sorted(fdf["Agency Short"].value_counts().head(8).index.tolist()),
        )

        if top_ag_range:
            sub = fdf[fdf["Agency Short"].isin(top_ag_range)].copy()
            sub = sub.sort_values("Salary Mid", ascending=True).head(80)

            fig = go.Figure()
            for _, row in sub.iterrows():
                fig.add_trace(go.Scatter(
                    x=[row["Salary Range From"], row["Salary Range To"]],
                    y=[row["Agency Short"], row["Agency Short"]],
                    mode="lines+markers",
                    line=dict(color=BOROUGH_COLORS.get(row["Borough"], "#1D4E89"), width=2),
                    marker=dict(size=[8, 8]),
                    hovertemplate=(
                        f"<b>{row['Business Title']}</b><br>"
                        f"From: ${row['Salary Range From']:,.0f}<br>"
                        f"To: ${row['Salary Range To']:,.0f}<br>"
                        f"Borough: {row['Borough']}<extra></extra>"
                    ),
                    showlegend=False,
                ))

            fig.update_layout(
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis_title="Annual Salary ($)", xaxis_tickformat="$,.0f",
                height=450,
                font=dict(family="Inter"),
                margin=dict(l=140, r=20, t=10, b=40),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
            st.plotly_chart(fig, use_container_width=True)

        # percentile breakdown
        st.markdown('<div class="section-header">Salary Percentile Breakdown</div>', unsafe_allow_html=True)
        pct_df = fdf.groupby("Borough")["Salary Mid"].quantile(
            [0.1, 0.25, 0.5, 0.75, 0.9]
        ).unstack().reset_index()
        pct_df.columns = ["Borough", "P10", "P25", "P50", "P75", "P90"]
        pct_df = pct_df[pct_df["Borough"] != "Unknown"]
        for c in ["P10","P25","P50","P75","P90"]:
            pct_df[c] = pct_df[c].apply(lambda x: f"${x:,.0f}")
        st.dataframe(pct_df, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — GEOGRAPHIC INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗺️ Geographic Insights":
    st.markdown("""
    <div class="page-header">
      <h1>🗺️ Geographic Insights</h1>
      <p>Where are NYC government jobs located? Explore by borough and work location.</p>
    </div>
    """, unsafe_allow_html=True)

    known = fdf[fdf["Borough"] != "Unknown"].copy()

    # ── choropleth map using plotly's built-in geojson ───────────────────────────
    st.markdown('<div class="section-header">Job Density & Salary by Borough</div>', unsafe_allow_html=True)

    map_metric = st.radio(
        "Map metric", ["Job Count", "Median Salary", "Avg # Positions"],
        horizontal=True
    )

    bor_stats = known.groupby("Borough").agg(
        job_count=("Job ID", "count"),
        median_salary=("Salary Mid", "median"),
        avg_positions=("# Of Positions", "mean"),
    ).reset_index()

    # Map borough names to NYC borough GeoJSON IDs
    borough_id_map = {
        "Manhattan": "New York",
        "Brooklyn": "Kings",
        "Queens": "Queens",
        "Bronx": "Bronx",
        "Staten Island": "Richmond",
    }
    bor_stats["county"] = bor_stats["Borough"].map(borough_id_map)

    metric_col = {"Job Count": "job_count",
                  "Median Salary": "median_salary",
                  "Avg # Positions": "avg_positions"}[map_metric]
    label_fmt = {"Job Count": ",.0f",
                 "Median Salary": "$,.0f",
                 "Avg # Positions": ".1f"}[map_metric]

    # Manual choropleth using borough polygons (approximate centroids + bubble map)
    # We use scatter_mapbox with borough centroids since full GeoJSON choropleth
    # requires the file; this is clean & interactive
    centroids = {
        "Manhattan":    (40.7831, -73.9712),
        "Brooklyn":     (40.6782, -73.9442),
        "Queens":       (40.7282, -73.7949),
        "Bronx":        (40.8448, -73.8648),
        "Staten Island":(40.5795, -74.1502),
    }
    bor_stats["lat"] = bor_stats["Borough"].map(lambda b: centroids.get(b,(0,0))[0])
    bor_stats["lon"] = bor_stats["Borough"].map(lambda b: centroids.get(b,(0,0))[1])

    fig_map = px.scatter_mapbox(
        bor_stats,
        lat="lat", lon="lon",
        size=metric_col,
        color=metric_col,
        color_continuous_scale=["#4CA3DD", "#0A2342"],
        hover_name="Borough",
        hover_data={
            "job_count": ":,.0f",
            "median_salary": ":$,.0f",
            "avg_positions": ":.1f",
            "lat": False, "lon": False,
        },
        size_max=80,
        zoom=9.5,
        center={"lat": 40.70, "lon": -73.94},
        mapbox_style="carto-positron",
        labels={
            "job_count": "Job Count",
            "median_salary": "Median Salary",
            "avg_positions": "Avg Positions",
        },
    )
    fig_map.update_layout(
        height=500,
        margin=dict(l=0, r=0, t=0, b=0),
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_map, use_container_width=True)

    # ── side-by-side bar charts ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Job Count by Borough</div>', unsafe_allow_html=True)
        bc = bor_stats.sort_values("job_count", ascending=False)
        fig = px.bar(bc, x="Borough", y="job_count",
                     color="Borough", color_discrete_map=BOROUGH_COLORS,
                     text="job_count")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", yaxis_title="# Postings",
                          height=320, font=dict(family="Inter"),
                          margin=dict(t=10, b=20))
        fig.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Median Salary by Borough</div>', unsafe_allow_html=True)
        bs = bor_stats.sort_values("median_salary", ascending=False)
        fig2 = px.bar(bs, x="Borough", y="median_salary",
                      color="Borough", color_discrete_map=BOROUGH_COLORS,
                      text="median_salary")
        fig2.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
        fig2.update_layout(showlegend=False, plot_bgcolor="white",
                           paper_bgcolor="white", yaxis_title="Median Salary ($)",
                           yaxis_tickformat="$,.0f",
                           height=320, font=dict(family="Inter"),
                           margin=dict(t=10, b=20))
        fig2.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig2, use_container_width=True)

    # ── heatmap: agency x borough ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Agency × Borough Heatmap (top 12 agencies)</div>', unsafe_allow_html=True)
    top12 = fdf["Agency Short"].value_counts().head(12).index
    heat_df = (known[known["Agency Short"].isin(top12)]
               .groupby(["Agency Short", "Borough"])["Job ID"]
               .count()
               .unstack(fill_value=0))
    fig_heat = px.imshow(
        heat_df,
        color_continuous_scale=["white", "#4CA3DD", "#0A2342"],
        text_auto=True,
        aspect="auto",
    )
    fig_heat.update_layout(
        height=380,
        font=dict(family="Inter"),
        coloraxis_showscale=True,
        margin=dict(l=160, r=20, t=10, b=60),
    )
    st.plotly_chart(fig_heat, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — TEXT & SKILLS ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📝 Text & Skills Analysis":
    st.markdown("""
    <div class="page-header">
      <h1>📝 Text & Skills Analysis</h1>
      <p>NLP-powered exploration of job descriptions, skills, and qualifications.</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["☁️ Word Clouds", "🔑 TF-IDF Keywords", "🗂️ Topic Modeling (NMF)"])

    sw = get_stopwords()

    # ── tab 1: word clouds ───────────────────────────────────────────────────────
    with tab1:
        wc_col1, wc_col2 = st.columns([1, 2])
        with wc_col1:
            wc_source = st.selectbox(
                "Text field",
                ["Job Description", "Preferred Skills", "Minimum Qual Requirements"],
            )
            wc_borough = st.selectbox("Filter by borough", ["All"] + sorted(
                [b for b in fdf["Borough"].unique() if b != "Unknown"]
            ))
            wc_n = st.slider("Max words", 50, 300, 150, step=10)

        sub_wc = fdf.copy()
        if wc_borough != "All":
            sub_wc = sub_wc[sub_wc["Borough"] == wc_borough]

        text_blob = " ".join(sub_wc[wc_source].dropna().tolist())
        if len(text_blob) < 100:
            st.warning("Not enough text for this selection.")
        else:
            # clean
            text_clean = re.sub(r"[^a-zA-Z\s]", " ", text_blob.lower())

            wc = WordCloud(
                width=900, height=500,
                background_color="white",
                stopwords=sw,
                max_words=wc_n,
                colormap="Blues",
                prefer_horizontal=0.8,
                collocations=False,
            ).generate(text_clean)

            fig_wc, ax = plt.subplots(figsize=(11, 5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            fig_wc.patch.set_facecolor("white")
            plt.tight_layout(pad=0)
            st.pyplot(fig_wc)

        # top word frequency bar
        st.markdown('<div class="section-header">Most Frequent Words</div>', unsafe_allow_html=True)
        words = [w for w in text_clean.split() if w not in sw and len(w) > 3]
        freq = Counter(words).most_common(20)
        freq_df = pd.DataFrame(freq, columns=["Word", "Count"])
        fig_freq = px.bar(
            freq_df, x="Count", y="Word", orientation="h",
            color="Count",
            color_continuous_scale=["#4CA3DD", "#0A2342"],
        )
        fig_freq.update_layout(
            showlegend=False, coloraxis_showscale=False,
            plot_bgcolor="white", paper_bgcolor="white",
            height=420, font=dict(family="Inter"),
            yaxis=dict(autorange="reversed"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_freq.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig_freq, use_container_width=True)

    # ── tab 2: TF-IDF by agency ──────────────────────────────────────────────────
    with tab2:
        st.markdown(
            "TF-IDF highlights terms that are **distinctive** to each agency's job postings "
            "— high TF-IDF means a word is frequent in that agency but rare elsewhere.",
        )

        tfidf_result = tfidf_by_category(fdf, col="Agency Short")

        selected_agencies = st.multiselect(
            "Select agencies to compare (2–6 recommended)",
            options=sorted(tfidf_result.keys()),
            default=list(sorted(tfidf_result.keys()))[:4],
        )

        if selected_agencies:
            n_ag = len(selected_agencies)
            cols = st.columns(min(n_ag, 3))
            for i, ag in enumerate(selected_agencies):
                with cols[i % 3]:
                    terms = tfidf_result.get(ag, [])
                    if not terms:
                        continue
                    df_t = pd.DataFrame(terms, columns=["Term", "TF-IDF"])
                    fig = px.bar(
                        df_t.sort_values("TF-IDF"),
                        x="TF-IDF", y="Term",
                        orientation="h",
                        title=ag,
                        color="TF-IDF",
                        color_continuous_scale=["#4CA3DD", "#0A2342"],
                    )
                    fig.update_layout(
                        showlegend=False, coloraxis_showscale=False,
                        plot_bgcolor="white", paper_bgcolor="white",
                        height=340, title_font_size=13,
                        font=dict(family="Inter", size=11),
                        margin=dict(l=5, r=5, t=36, b=10),
                    )
                    fig.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
                    st.plotly_chart(fig, use_container_width=True)

    # ── tab 3: topic modeling ────────────────────────────────────────────────────
    with tab3:
        st.markdown(
            "**NMF topic modeling** on job descriptions identifies latent skill themes "
            "across all NYC government postings."
        )

        n_topics = st.slider("Number of topics", 4, 12, 8)
        text_source = st.radio("Source text", ["Job Description", "Preferred Skills"], horizontal=True)

        texts = fdf[text_source].dropna().tolist()

        with st.spinner("Running topic model…"):
            W, topics = run_topic_model(texts, n_topics=n_topics)

        # give each topic a readable name from its top term
        topic_names = {k: k + f" ({v[0][0].title()})" for k, v in topics.items()}

        # visualise top words per topic as a heatmap
        topic_labels = list(topics.keys())
        all_terms = []
        for terms in topics.values():
            for t, _ in terms:
                if t not in all_terms:
                    all_terms.append(t)
        all_terms = all_terms[:30]

        matrix = []
        for tk in topic_labels:
            row = {t: 0.0 for t in all_terms}
            for term, score in topics[tk]:
                if term in row:
                    row[term] = score
            matrix.append([row[t] for t in all_terms])

        fig_hm = px.imshow(
            matrix,
            x=all_terms,
            y=[topic_names[t] for t in topic_labels],
            color_continuous_scale=["white", "#4CA3DD", "#0A2342"],
            aspect="auto",
        )
        fig_hm.update_layout(
            height=420,
            font=dict(family="Inter", size=11),
            margin=dict(l=10, r=10, t=10, b=80),
        )
        fig_hm.update_xaxes(tickangle=45)
        st.plotly_chart(fig_hm, use_container_width=True)

        # topic strength distribution
        st.markdown('<div class="section-header">Topic Strength Across Postings</div>', unsafe_allow_html=True)
        topic_strengths = W.mean(axis=0)
        strength_df = pd.DataFrame({
            "Topic": [topic_names[t] for t in topic_labels],
            "Avg Strength": topic_strengths,
        }).sort_values("Avg Strength", ascending=False)

        fig_bar = px.bar(
            strength_df,
            x="Avg Strength", y="Topic",
            orientation="h",
            color="Avg Strength",
            color_continuous_scale=["#4CA3DD", "#0A2342"],
        )
        fig_bar.update_layout(
            coloraxis_showscale=False, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            height=350, font=dict(family="Inter"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_bar.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig_bar, use_container_width=True)

        # individual topic deep-dives
        st.markdown('<div class="section-header">Topic Deep-Dive</div>', unsafe_allow_html=True)
        sel_topic = st.selectbox("Select a topic", list(topic_names.values()))
        sel_key = [k for k, v in topic_names.items() if v == sel_topic][0]
        terms_df = pd.DataFrame(topics[sel_key], columns=["Term", "Score"])
        fig_td = px.bar(
            terms_df.sort_values("Score"),
            x="Score", y="Term", orientation="h",
            color="Score",
            color_continuous_scale=["#4CA3DD", "#0A2342"],
        )
        fig_td.update_layout(
            coloraxis_showscale=False, showlegend=False,
            plot_bgcolor="white", paper_bgcolor="white",
            height=380, font=dict(family="Inter"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        fig_td.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig_td, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — AGENCY DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏢 Agency Deep-Dive":
    st.markdown("""
    <div class="page-header">
      <h1>🏢 Agency Deep-Dive</h1>
      <p>Drill into any NYC agency to explore its roles, salaries, and geographic footprint.</p>
    </div>
    """, unsafe_allow_html=True)

    agencies = sorted(fdf["Agency Short"].value_counts().index.tolist())
    sel_ag = st.selectbox("Select an agency", agencies)

    ag_df = fdf[fdf["Agency Short"] == sel_ag].copy()

    # ── KPIs ─────────────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Postings", f"{len(ag_df):,}")
    c2.metric("Total Positions", f"{ag_df['# Of Positions'].sum():,}")
    c3.metric("Median Salary", f"${ag_df['Salary Mid'].median():,.0f}")
    c4.metric("Salary Range",
              f"${ag_df['Salary Range From'].min():,.0f}–${ag_df['Salary Range To'].max():,.0f}")

    col_l, col_r = st.columns(2)

    # ── salary distribution ──────────────────────────────────────────────────────
    with col_l:
        st.markdown('<div class="section-header">Salary Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ag_df["Salary Mid"],
            nbinsx=25,
            marker_color="#1D4E89",
            opacity=0.85,
        ))
        fig.add_vline(
            x=ag_df["Salary Mid"].median(),
            line_dash="dash", line_color="#F7B731",
            annotation_text=f"Median: ${ag_df['Salary Mid'].median():,.0f}",
            annotation_position="top right",
            annotation_font_color="#F7B731",
        )
        fig.update_layout(
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis_title="Salary ($)", yaxis_title="Count",
            xaxis_tickformat="$,.0f",
            height=320, font=dict(family="Inter"),
            margin=dict(t=10, b=10),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
        fig.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
        st.plotly_chart(fig, use_container_width=True)

    # ── posting type breakdown ───────────────────────────────────────────────────
    with col_r:
        st.markdown('<div class="section-header">Posting Type & Borough</div>', unsafe_allow_html=True)
        breakdown = ag_df.groupby(["Posting Type", "Borough"])["Job ID"].count().reset_index()
        breakdown.columns = ["Posting Type", "Borough", "Count"]
        fig2 = px.sunburst(
            breakdown,
            path=["Posting Type", "Borough"],
            values="Count",
            color="Posting Type",
            color_discrete_map={"External": "#1D4E89", "Internal": "#F7B731"},
        )
        fig2.update_layout(
            height=320,
            font=dict(family="Inter"),
            margin=dict(t=10, b=10, l=10, r=10),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── top roles table ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Top Job Titles</div>', unsafe_allow_html=True)
    top_roles = (ag_df.groupby("Business Title")
                 .agg(count=("Job ID","count"),
                      median_salary=("Salary Mid","median"),
                      total_positions=("# Of Positions","sum"))
                 .reset_index()
                 .sort_values("count", ascending=False)
                 .head(20))
    top_roles["median_salary"] = top_roles["median_salary"].apply(lambda x: f"${x:,.0f}")
    top_roles.columns = ["Business Title", "# Postings", "Median Salary", "Total Positions"]
    st.dataframe(top_roles, use_container_width=True, hide_index=True)

    # ── salary vs. positions scatter ─────────────────────────────────────────────
    st.markdown('<div class="section-header">Salary vs. Number of Positions</div>', unsafe_allow_html=True)
    fig3 = px.scatter(
        ag_df,
        x="Salary Mid",
        y="# Of Positions",
        color="Borough",
        color_discrete_map=BOROUGH_COLORS,
        hover_data=["Business Title", "Posting Type"],
        size="Salary Range To",
        size_max=20,
        opacity=0.75,
    )
    fig3.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis_title="Midpoint Salary ($)", yaxis_title="# Positions",
        xaxis_tickformat="$,.0f",
        height=360, font=dict(family="Inter"),
        margin=dict(t=10, b=20),
    )
    fig3.update_xaxes(showgrid=True, gridcolor="#DCE3ED")
    fig3.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
    st.plotly_chart(fig3, use_container_width=True)

    # ── skills word cloud for this agency ────────────────────────────────────────
    st.markdown('<div class="section-header">Top Skills & Keywords (this agency)</div>', unsafe_allow_html=True)
    sw = get_stopwords()
    blob = " ".join(ag_df["Job Description"].dropna().tolist())
    blob_clean = re.sub(r"[^a-zA-Z\s]", " ", blob.lower())
    words = [w for w in blob_clean.split() if w not in sw and len(w) > 3]
    freq = Counter(words).most_common(25)
    freq_df = pd.DataFrame(freq, columns=["Word", "Frequency"])
    fig_wf = px.bar(
        freq_df, x="Word", y="Frequency",
        color="Frequency",
        color_continuous_scale=["#4CA3DD", "#0A2342"],
    )
    fig_wf.update_layout(
        coloraxis_showscale=False,
        plot_bgcolor="white", paper_bgcolor="white",
        height=320, font=dict(family="Inter"),
        margin=dict(t=10, b=40),
        xaxis_tickangle=45,
    )
    fig_wf.update_yaxes(showgrid=True, gridcolor="#DCE3ED")
    st.plotly_chart(fig_wf, use_container_width=True)
