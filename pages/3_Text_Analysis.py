import plotly.express as px
import streamlit as st

from utils import (
    load_postings,
    apply_sidebar_filters,
    sample_for_text,
    get_ngram_counts,
    get_tfidf_by_group,
    clean_plot,
    note,
)

st.set_page_config(page_title="Text Analysis", page_icon="🧠", layout="wide")

# ── Load and filter data ────────────────────────────────────────────────────
df = load_postings()
filtered = apply_sidebar_filters(df)
text_df = sample_for_text(filtered, max_rows=8000)

st.title("🧠 Skills and Text Analysis")
st.markdown("This page uses simple NLP methods to summarize what employers emphasize in job postings.")

st.caption(f"Rows used for text analysis: {len(text_df):,}. A sample is used because the filtered data is large so the app stays responsive.")
st.divider()

# ── Common terms and bigrams ────────────────────────────────────────────────
st.subheader("Most common language in postings")
text_source = st.selectbox("Text source", ["text_full", "description", "skills_desc"])

left, right = st.columns(2)

with left:
    unigram_df = get_ngram_counts(text_df, text_col=text_source, ngram_range=(1, 1), top_n=20)
    fig_uni = px.bar(
        unigram_df.sort_values("count"),
        x="count",
        y="term",
        orientation="h",
        title="Most common words",
        labels={"count": "Count", "term": "Word"},
    )
    st.plotly_chart(clean_plot(fig_uni, height=520), width="stretch")

with right:
    bigram_df = get_ngram_counts(text_df, text_col=text_source, ngram_range=(2, 2), top_n=20)
    fig_bi = px.bar(
        bigram_df.sort_values("count"),
        x="count",
        y="term",
        orientation="h",
        title="Most common two-word phrases",
        labels={"count": "Count", "term": "Bigram"},
    )
    st.plotly_chart(clean_plot(fig_bi, height=520), width="stretch")

note(
    "The common-word view gives a broad sense of what the postings talk about most often. Bigrams are easier "
    "to interpret because phrases like tools, soft skills, or job functions keep more context than single words."
)

st.divider()

# ── TF-IDF by group ─────────────────────────────────────────────────────────
st.subheader("Distinctive terms by group")
group_choice = st.selectbox("Compare groups by", ["work_type_clean", "experience_clean", "state"])

tfidf_df = get_tfidf_by_group(text_df, group_col=group_choice, top_n=8)

if tfidf_df.empty:
    st.warning("There is not enough text to calculate TF-IDF.")
else:
    groups = sorted(tfidf_df[group_choice].dropna().unique().tolist())
    default_groups = groups[:3]
    selected_groups = st.multiselect("Groups to show", groups, default=default_groups)

    plot_df = tfidf_df[tfidf_df[group_choice].isin(selected_groups)]

    if plot_df.empty:
        st.warning("Choose at least one group to display.")
    else:
        fig_tfidf = px.bar(
            plot_df,
            x="score",
            y="term",
            color=group_choice,
            facet_col=group_choice,
            facet_col_wrap=2,
            orientation="h",
            title="TF-IDF terms by selected group",
            labels={"score": "TF-IDF score", "term": "Term"},
        )
        fig_tfidf.update_yaxes(matches=None, showticklabels=True)
        fig_tfidf.update_layout(showlegend=False)
        st.plotly_chart(clean_plot(fig_tfidf, height=760), width="stretch")

note(
    "TF-IDF gives which terms help distinguish one group from another. That makes the results more useful for comparing "
    "different work types, experience levels, or states."
)


st.divider()