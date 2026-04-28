import streamlit as st
import plotly.express as px

from utils import load_postings, apply_sidebar_filters, clean_plot, note

st.set_page_config(page_title="National Job Market Explorer", page_icon="🏠", layout="wide")

# ── Data  ────────────────────────────────────────────────────────────
df = load_postings()
filtered = apply_sidebar_filters(df)

# ── Title & project introduction ─────────────────────────────────────────────────
st.title("🏠 National Job Market Explorer")
st.markdown(
    "This dashboard explores a national job-postings dataset through three angles: "
    "**market overview**, **geographic patterns**, and **skill language**. "
    "The main question is simple: where are the opportunities, what do they pay, "
    "and what are employers asking for?"
)

# ── Sections ────────────────────────────────────────────────────────
st.subheader("Explore the project")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Data Overview")
    st.markdown("Look at posting volume, salary distributions, and work arrangements.")
    st.page_link("pages/1_Data_Overview.py", label="Go to Data Overview")

with col2:
    st.markdown("### 🗺️ Geographic Analysis")
    st.markdown("Compare state-level posting counts, salaries, and remote-work share.")
    st.page_link("pages/2_Geographic_Analysis.py", label="Go to Geographic Analysis")

with col3:
    st.markdown("### 🧠 Text Analysis")
    st.markdown("Use n-grams and TF-IDF to summarize job-description language.")
    st.page_link("pages/3_Text_Analysis.py", label="Go to Text Analysis")

st.divider()

# ── First visual summary ────────────────────────────────────────────────────
st.subheader("A quick first look")
state_counts = (
    filtered.dropna(subset=["state"])
    .groupby("state")
    .size()
    .reset_index(name="postings")
    .sort_values("postings", ascending=False)
    .head(15)
)

fig = px.bar(
    state_counts.sort_values("postings"),
    x="postings",
    y="state",
    orientation="h",
    title="Top 15 states by posting volume",
    labels={"postings": "Number of postings", "state": "State"},
)
st.plotly_chart(clean_plot(fig, height=430), width="stretch")

note(
    "This first chart is only descriptive, but it helps orient the reader before the more detailed pages. "
    "A state with many postings is not automatically the best labor market andthat is why the next pages "
    "also compare salary, remote share, and skill language."
)
