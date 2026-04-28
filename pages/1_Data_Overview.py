import plotly.express as px
import streamlit as st

from utils import load_postings, apply_sidebar_filters, clean_plot, note

st.set_page_config(page_title="Data Overview", page_icon="📊", layout="wide")

# ── Load and filter data ────────────────────────────────────────────────────
df = load_postings()
filtered = apply_sidebar_filters(df)

st.title("📊 Data Overview")
st.markdown("This page gives a basic descriptive view of the filtered job market.")

note(
    "We start with simple EDA because it gives the rest of the project context. Before mapping or NLP, "
    "we need to know the size of the data, how much salary information is available, and what kinds of "
    "work arrangements dominate the postings."
)

st.divider()

# ── Summary metrics ─────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric("Postings", f"{len(filtered):,}")
c2.metric("Unique job titles", f"{filtered['title_clean'].nunique():,}")
c3.metric("Salary available", f"{filtered['salary_mid'].notna().mean() * 100:.1f}%")
c4.metric("Remote share", f"{filtered['remote_flag'].mean() * 100:.1f}%")

st.divider()

# ── Salary and work type charts ─────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.subheader("Salary distribution")
    salary_df = filtered.dropna(subset=["salary_mid"])
    fig_salary = px.histogram(
        salary_df,
        x="salary_mid",
        nbins=50,
        title="Distribution of annualized salary",
        labels={"salary_mid": "Annualized salary ($)", "count": "Postings"},
    )
    fig_salary.update_xaxes(tickformat="$,.0f")
    st.plotly_chart(clean_plot(fig_salary), width="stretch")

with right:
    st.subheader("Work type mix")
    work_counts = filtered["work_type_clean"].value_counts().reset_index()
    work_counts.columns = ["work_type", "postings"]
    fig_work = px.bar(
        work_counts.sort_values("postings"),
        x="postings",
        y="work_type",
        orientation="h",
        title="Postings by work type",
        labels={"postings": "Number of postings", "work_type": "Work type"},
    )
    st.plotly_chart(clean_plot(fig_work), width="stretch")

note(
    "The salary chart should be read carefully because not every posting reports pay. I kept the salary-coverage "
    "metric visible so the audience can see this limitation instead of assuming the salary results represent every job."
)

st.divider()

# ── Salary by experience level ──────────────────────────────────────────────
st.subheader("Salary by experience level")
salary_exp = filtered.dropna(subset=["salary_mid"])
common_exp = salary_exp["experience_clean"].value_counts().head(8).index
salary_exp = salary_exp[salary_exp["experience_clean"].isin(common_exp)]

fig_box = px.box(
    salary_exp,
    x="experience_clean",
    y="salary_mid",
    points=False,
    title="Annualized salary across experience levels",
    labels={"experience_clean": "Experience level", "salary_mid": "Annualized salary ($)"},
)
fig_box.update_yaxes(tickformat="$,.0f")
st.plotly_chart(clean_plot(fig_box, height=470), width="stretch")

note(
    "This chart connects directly to the job-search question. Experience level is one of the clearest structured "
    "signals in the dataset, and the boxplot shows both typical salary differences and variation within each group."
)

# ── Raw data preview ────────────────────────────────────────────────────────
with st.expander("View a small sample of the cleaned data"):
    st.dataframe(
        filtered[["title_clean", "state", "work_type_clean", "experience_clean", "remote_flag", "salary_mid"]].head(100),
        width="stretch",
    )
