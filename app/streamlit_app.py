"""TriageLens demo: predict the type of a GitHub issue and find possible duplicates.

Run locally:
    pip install -r requirements.txt
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import os

import streamlit as st

from triage_lens.app_core import DemoAssets, load_encoder, plain_snippet

HF_DATASET = os.environ.get("TRIAGE_LENS_DATASET", "IlyaShaposhnikov/triage-lens-demo")
GITHUB = "https://github.com/IlyaShaposhnikov/triage-lens"
CLASS_NAMES = {"bug": "Bug", "feature": "Feature request", "question": "Question", "docs": "Docs"}

# Written for the demo; not copied from either issue tracker.
EXAMPLES = {
    "astral-sh/uv": {
        "title": "uv sync fails behind a corporate proxy with a self-signed certificate",
        "body": "Running `uv sync` in our CI fails with `invalid peer certificate: "
        "UnknownIssuer`. pip works with the same proxy because it reads REQUESTS_CA_BUNDLE. "
        "Is there an equivalent setting for uv?",
    },
    "pandas-dev/pandas": {
        "title": "read_csv silently converts large integer IDs to float",
        "body": "When a column of 19-digit IDs has a missing value, read_csv turns the whole "
        "column into float64 and the IDs lose precision. I expected a nullable integer dtype "
        "or at least a warning.",
    },
}

st.set_page_config(page_title="TriageLens", page_icon="🔎", layout="wide")


# The index is refreshed on a schedule (triage_lens.refresh); after the TTL the
# next visitor triggers a check. An unchanged dataset costs one metadata request,
# since huggingface_hub keeps the files in its local cache.
INDEX_TTL_S = 6 * 3600


@st.cache_resource(ttl=INDEX_TTL_S, show_spinner="Downloading the index from Hugging Face…")
def get_assets() -> DemoAssets:
    return DemoAssets.from_hub(HF_DATASET)


@st.cache_resource(show_spinner="Loading the embedding model…")
def get_encoder(name: str):
    return load_encoder(name)


try:
    assets = get_assets()
    encode = get_encoder(assets.encoder_name)
except Exception as exc:  # network, missing dataset, incompatible files
    st.error(
        f"Could not load the demo data from Hugging Face (`{HF_DATASET}`): {exc}. "
        "Reload the page in a minute; if it persists, the dataset may be unavailable."
    )
    st.stop()

with st.sidebar:
    st.header("TriageLens")
    repo = st.selectbox("Repository", list(assets.repos))
    k = st.slider("Similar issues to show", 3, 15, 5)
    info = assets.repos[repo].info
    refreshed = info.get("refreshed_at")
    st.caption(
        f"{info['issues']:,} issues indexed, newest from {info['newest_issue'][:10]}"
        + (f", refreshed {refreshed[:10]}" if refreshed else "")
        + f". Encoder: `{assets.encoder_name}`."
    )
    metrics = info.get("metrics", {})
    if "search" in metrics:
        st.caption(
            f"On past duplicates, the original is in the top 10 for "
            f"{metrics['search']['recall@10']:.0%} of them."
        )
    if "classifier_test" in metrics:
        st.caption(
            f"Issue type accuracy on recent issues: {metrics['classifier_test']['accuracy']:.0%}."
        )
    st.markdown(f"[How it was built and evaluated]({GITHUB})")

st.title("🔎 TriageLens")
st.write(
    "Paste a new GitHub issue. The app predicts what kind of issue it is and lists earlier "
    "issues that may describe the same problem."
)

if st.button("Fill in an example"):
    st.session_state.title = EXAMPLES[repo]["title"]
    st.session_state.body = EXAMPLES[repo]["body"]

title = st.text_input("Title", key="title", placeholder="What went wrong or what do you need?")
body = st.text_area(
    "Description",
    key="body",
    height=180,
    placeholder="Steps, error messages, what you expected…",
)

if st.button("Triage", type="primary", disabled=not (title or body).strip()):
    with st.spinner("Looking through earlier issues…"):
        probs = assets.classify(repo, title, body)
        results = assets.similar(repo, title, body, encode, k=k)

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Issue type")
        best = max(probs, key=probs.get)
        st.markdown(f"Most likely: **{CLASS_NAMES.get(best, best)}** ({probs[best]:.0%})")
        for cls, p in sorted(probs.items(), key=lambda item: -item[1]):
            st.progress(p, text=f"{CLASS_NAMES.get(cls, cls)} · {p:.0%}")
        st.caption(
            "TF-IDF + logistic regression. On recent issues it agrees with the maintainers' "
            "label about 70–80% of the time; treat it as a hint, not a verdict."
        )
    with right:
        st.subheader("Possibly related issues")
        for issue in results:
            state = issue["state"]
            if issue.get("state_reason") == "duplicate":
                state += " · duplicate"
            labels = ", ".join(issue["labels"][:4])
            title_md = issue["title"].replace("[", "\\[").replace("]", "\\]")
            st.markdown(
                f"**[#{issue['number']} {title_md}]({issue['url']})**  \n"
                f"similarity {issue['similarity']:.2f} · {state} · {issue['created_at'][:10]}"
                + (f" · {labels}" if labels else "")
            )
            snippet = plain_snippet(issue["snippet"])
            if snippet:
                st.caption(snippet)
