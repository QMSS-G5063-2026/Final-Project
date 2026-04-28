from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS

# These are the only columns we need and used for the dashboard.
USECOLS = [
    "company_name", "title", "description", "max_salary", "min_salary",
    "normalized_salary", "location", "remote_allowed", "formatted_work_type",
    "formatted_experience_level", "skills_desc", "zip_code"
]


@st.cache_data(show_spinner=False)
def load_postings():
    """Load and clean the job postings data.

    We only create the variables that are useful for the charts: state, salary midpoint, remote flag, clean labels,
    and one combined text column for the NLP section.
    """
    base = Path(__file__).resolve().parent
    data_path = base / "data" / "postings_sample.csv" 

    df = pd.read_csv(data_path, usecols=lambda c: c in USECOLS)

    # Pull the state abbreviation from locations like "New York, NY".
    df["state"] = df["location"].astype(str).str.extract(r",\s*([A-Z]{2})\s*$")[0]

    # Convert salary columns to numeric before making salary_mid.
    for col in ["normalized_salary", "min_salary", "max_salary"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove extreme values that are probably hourly, missing, or data errors.
    df.loc[(df["normalized_salary"] < 10000) | (df["normalized_salary"] > 500000), "normalized_salary"] = np.nan
    df.loc[(df["min_salary"] < 10000) | (df["min_salary"] > 500000), "min_salary"] = np.nan
    df.loc[(df["max_salary"] < 10000) | (df["max_salary"] > 800000), "max_salary"] = np.nan

    # Prefer normalized salary when it exists. If not, use the midpoint of min/max.
    df["salary_mid"] = df["normalized_salary"]
    need_mid = df["salary_mid"].isna() & df["min_salary"].notna() & df["max_salary"].notna()
    df.loc[need_mid, "salary_mid"] = (df.loc[need_mid, "min_salary"] + df.loc[need_mid, "max_salary"]) / 2

    df["remote_flag"] = df["remote_allowed"].fillna(False).astype(bool)
    df["title_clean"] = df["title"].fillna("Unknown title")
    df["work_type_clean"] = df["formatted_work_type"].fillna("Unknown")
    df["experience_clean"] = df["formatted_experience_level"].fillna("Unknown")

    # Create a combined text field lets us reuse the same NLP functions for descriptions and skills
    df["text_full"] = (
        df["title_clean"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["skills_desc"].fillna("")
    ).str.replace(r"\s+", " ", regex=True).str.strip()

    return df


def apply_sidebar_filters(df):
    """Shared filters used across the pages."""
    with st.sidebar:
        st.header("Filters")

        states = sorted(df["state"].dropna().unique().tolist())
        selected_states = st.multiselect("State", states, default=[])

        work_types = sorted(df["work_type_clean"].dropna().unique().tolist())
        selected_work = st.multiselect("Work type", work_types, default=[])

        exp_levels = sorted(df["experience_clean"].dropna().unique().tolist())
        selected_exp = st.multiselect("Experience level", exp_levels, default=[])

        remote_choice = st.selectbox(
            "Remote filter",
            ["All postings", "Remote only", "On-site / unspecified only"],
        )

        salary_data = df["salary_mid"].dropna()
        if len(salary_data) > 0:
            low = int(max(10000, salary_data.quantile(0.01)))
            high = int(min(300000, salary_data.quantile(0.99)))

        salary_range = st.slider(
            "Annualized salary range ($)",
            min_value=low,
            max_value=high,
            value=(low, high),
            step=5000,
        )

    filtered = df.copy()

    if selected_states:
        filtered = filtered[filtered["state"].isin(selected_states)]
    if selected_work:
        filtered = filtered[filtered["work_type_clean"].isin(selected_work)]
    if selected_exp:
        filtered = filtered[filtered["experience_clean"].isin(selected_exp)]

    if remote_choice == "Remote only":
        filtered = filtered[filtered["remote_flag"]]
    elif remote_choice == "On-site / unspecified only":
        filtered = filtered[~filtered["remote_flag"]]

    # Keep rows with missing salary because those postings still matter for counts
    # and text analysis. Only filter the rows where salary is available.
    filtered = filtered[
        filtered["salary_mid"].isna() |
        filtered["salary_mid"].between(salary_range[0], salary_range[1])
    ]

    return filtered

@st.cache_data(show_spinner=False)
def sample_for_text(df, max_rows=8000):
    """Use a sample for NLP so the app does not become very slow."""
    if len(df) <= max_rows:
        return df.copy()
    return df.sample(n=max_rows, random_state=22)


def stop_words():
    """Small custom stopword list for job-posting text.

    These words are common in postings but not very meaningful for our research
    question. This mirrors what we did in the NLP assignment: remove words that
    dominate the text but do not tell us much substantively.
    """
    return list(ENGLISH_STOP_WORDS.union({
        "the", "and", "for", "with", "that", "this", "you", "your", "will", "our",
        "are", "have", "has", "from", "but", "all", "job", "jobs", "role", "work",
        "to", "of", "in", "is", "as", "we", "on", "at", "by", "be", "any", "re",
        "working", "team", "teams", "ability", "experience", "required", "preferred",
        "skills", "skill", "including", "years", "year", "must", "new", "york", "city",
        "position", "candidate", "candidates", "qualified", "equal", "opportunity",
        "employer", "apply", "using", "use", "support", "services", "service", "strong",
        "knowledge", "business", "provide", "within", "across", "looking", "seeking",
        "please", "full", "time", "responsible", "responsibilities", "company",
    }))


@st.cache_data(show_spinner=False)
def get_ngram_counts(df, text_col="text_full", ngram_range=(1, 1), top_n=20):
    """Count common words or phrases in the selected postings."""
    texts = df[text_col].dropna().astype(str)
    texts = texts[texts.str.len() > 30]

    if texts.empty:
        return pd.DataFrame(columns=["term", "count"])

    vectorizer = CountVectorizer(
        stop_words=stop_words(),
        lowercase=True,
        ngram_range=ngram_range,
        max_features=5000,
    )

    X = vectorizer.fit_transform(texts)
    counts = np.asarray(X.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()

    out = pd.DataFrame({"term": terms, "count": counts})
    return out.sort_values("count", ascending=False).head(top_n)


@st.cache_data(show_spinner=False)
def get_tfidf_by_group(df, group_col, top_n=8):
    """Find terms that are distinctive for each group.

    TF-IDF is useful here because it does not only reward common words. It gives
    more weight to words that are common inside one group but less common across
    the whole corpus.
    """
    grouped = (
        df[[group_col, "text_full"]]
        .dropna()
        .groupby(group_col)["text_full"]
        .apply(lambda s: " ".join(s.astype(str)))
        .reset_index()
    )

    # remove groups with very little text
    grouped = grouped[grouped["text_full"].str.len() > 300]

    vectorizer = TfidfVectorizer(
        stop_words=stop_words(),
        lowercase=True,
        ngram_range=(1, 2),
        max_features=3500,
        min_df=2,
        max_df=0.9,
    )

    X = vectorizer.fit_transform(grouped["text_full"])
    terms = vectorizer.get_feature_names_out()

    rows = []
    for i, group_name in enumerate(grouped[group_col]):
        scores = X[i].toarray().ravel()
        top_idx = scores.argsort()[-top_n:][::-1]
        for j in top_idx:
            rows.append({group_col: group_name, "term": terms[j], "score": scores[j]})

    return pd.DataFrame(rows)

def clean_plot(fig, height=420):
    """Use the same visual style across pages."""
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=10, r=10, t=45, b=20),
        font=dict(size=13),
    )
    return fig


def note(text):
    """Small narrative box used throughout the app."""
    st.markdown(
        f"""
        <div style="background:skyblue;border:1px solid #DCE3ED;border-radius:10px;
                    padding:14px 16px;margin:10px 0 18px 0;line-height:1.5;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )
