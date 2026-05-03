import plotly.express as px
import streamlit as st
import altair as alt

from utils import load_postings, apply_sidebar_filters, clean_plot, note

st.set_page_config(page_title="Geographic Analysis", page_icon="🗺️", layout="wide")

# ── Load and filter data ────────────────────────────────────────────────────
df = load_postings()
filtered = apply_sidebar_filters(df)

st.title("🗺️ Geographic Analysis")
st.markdown("This page uses state-level aggregation to compare job demand, salary, and remote-work flexibility.")

note(
    "For the map section, we extract state from the location field, group postings by state, "
    "and visualize those state summaries."
)

st.divider()

# ── State-level aggregation ─────────────────────────────────────────────────
geo_df = (
    filtered.dropna(subset=["state"])
    .groupby("state")
    .agg(
        postings=("title_clean", "count"),
        median_salary=("salary_mid", "median"),
        remote_share=("remote_flag", "mean"),
        salary_coverage=("salary_mid", lambda s: s.notna().mean()),
    )
    .reset_index()
)

# Very small state groups can make salary patterns noisy, so we keep a light cutoff.
metric_choice = st.selectbox(
    "Choose the map metric",
    ["Posting count", "Median salary", "Remote share", "Salary coverage"],
)

metric_map = {
    "Posting count": "postings",
    "Median salary": "median_salary",
    "Remote share": "remote_share",
    "Salary coverage": "salary_coverage",
}
color_col = metric_map[metric_choice]

# ── Choropleth map ──────────────────────────────────────────────────────────
fig_map = px.choropleth(
    geo_df,
    locations="state",
    locationmode="USA-states",
    scope="usa",
    color=color_col,
    hover_name="state",
    hover_data={
        "postings": ":,",
        "median_salary": ":$,.0f",
        "remote_share": ":.1%",
        "salary_coverage": ":.1%",
    },
    title=f"{metric_choice} by state",
    color_continuous_scale="Blues",
)
fig_map.update_layout(height=560)
st.plotly_chart(fig_map, width="stretch")

note(
    "Each state is colored based on the selected metric. This allows us to compare different aspects of the job market, "
    "such as posting volume, salary, or remote share, without changing the underlying data."
    "Large states like California, Texas, and New York often dominate in posting volume, but when switching to salary "
    "or remote share, the pattern changes. This suggests that job availability and job quality are not evenly distributed across the country."
)
st.divider()

# ── Linked state comparison ─────────────────────────────────────────────────
st.subheader("Linked state comparison")

note(
    "This view works like the linked-selection charts we practiced in class. "
    "Drag over states in the bar chart, and the scatterplot will highlight the same selected states. "
    "This makes it easier to connect state ranking with salary and remote-work patterns."
)

linked_df = geo_df.dropna(subset=["median_salary"]).copy()

brush = alt.selection_interval(encodings=["y"])

bar = (
    alt.Chart(linked_df)
    .mark_bar()
    .encode(
        x=alt.X("postings:Q", title="Number of postings"),
        y=alt.Y("state:N", sort="-x", title="State"),
        color=alt.condition(brush, alt.value("#1f77b4"), alt.value("lightgray")),
        tooltip=[
            alt.Tooltip("state:N", title="State"),
            alt.Tooltip("postings:Q", title="Postings", format=","),
            alt.Tooltip("median_salary:Q", title="Median salary", format="$,.0f"),
            alt.Tooltip("remote_share:Q", title="Remote share", format=".1%"),
        ],
    )
    .add_params(brush)
    .properties(height=420, title="Select states by posting volume")
)

scatter = (
    alt.Chart(linked_df)
    .mark_circle(size=90, opacity=0.75)
    .encode(
        x=alt.X("remote_share:Q", title="Remote share", axis=alt.Axis(format="%")),
        y=alt.Y("median_salary:Q", title="Median salary", axis=alt.Axis(format="$,.0f")),
        size=alt.Size("postings:Q", title="Postings"),
        color=alt.condition(brush, alt.value("#1f77b4"), alt.value("lightgray")),
        tooltip=[
            alt.Tooltip("state:N", title="State"),
            alt.Tooltip("postings:Q", title="Postings", format=","),
            alt.Tooltip("median_salary:Q", title="Median salary", format="$,.0f"),
            alt.Tooltip("remote_share:Q", title="Remote share", format=".1%"),
        ],
    )
    .properties(height=420, title="Selected states: salary vs. remote share")
)

linked_chart = alt.hconcat(bar, scatter).resolve_scale(color="independent")

st.altair_chart(linked_chart, width="stretch")

note(
    "The bar chart shows where postings are concentrated, while the scatterplot shows whether those states also have higher salary "
    "or more remote flexibility. This is useful because a state with many postings is not always the strongest state on compensation."
)

# ── Work type mix by state ──────────────────────────────────────────────────
st.subheader("Work type mix by state")

note(
    "This heatmap compares job structure across the most active states. It adds detail beyond the map because two states "
    "can have similar posting volume but very different mixes of full-time, contract, internship, or part-time roles."
)

top_states_heat = filtered["state"].value_counts().head(12).index.tolist()
top_work_types = filtered["work_type_clean"].value_counts().head(6).index.tolist()

heat_df = (
    filtered[
        filtered["state"].isin(top_states_heat)
        & filtered["work_type_clean"].isin(top_work_types)
    ]
    .groupby(["state", "work_type_clean"])
    .size()
    .reset_index(name="count")
)

heat_pivot = heat_df.pivot(
    index="state",
    columns="work_type_clean",
    values="count"
).fillna(0)

fig_heat = px.imshow(
    heat_pivot,
    text_auto=True,
    aspect="auto",
    color_continuous_scale="Blues",
    labels=dict(x="Work type", y="State", color="Postings"),
    title="Work type distribution across top states",
)

st.plotly_chart(clean_plot(fig_heat, height=500), width="stretch")

note(
    "Darker cells mean more postings in that state-work type combination. This makes it easier to see whether the job market "
    "in a state is mostly full-time roles or whether it has more variation across work arrangements."
)
