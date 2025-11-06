# streamlit_app.py
# ---------------------------------------------------------------
# RivalLens Mini — SEO Automation Flow (Beautiful Streamlit UI)
# ---------------------------------------------------------------

import os
import re
import json
import time
import html
import textwrap
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import streamlit as st

# -------------------------
# Helpers & Config
# -------------------------
APP_TITLE = "RivalLens Mini — SEO Automation"
APP_SUBTITLE = "URLs → Keywords → User Questions → AI → Meta/Headings → Internal Links"
MAX_URLS = 10

DEFAULT_FAQ_COUNT = 5
USER_QUESTION_SOURCES = ["reddit.com", "quora.com"]

STOPWORDS = set(
    """
    a about above after again against all am an and any are aren't as at be because been before being below between both
    but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further had
    hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd
    i'll i'm i've if in into is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once
    only or other ought our ours ourselves out over own same shan't she she'd she'll she's should shouldn't so some such
    than that that's the their theirs them themselves then there there's these they they'd they'll they're they've this
    those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where
    where's which while who who's whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself
    yourselves
    """.split()
)

# -------------------------
# Dataclasses
# -------------------------
@dataclass
class PageData:
    url: str
    title: str
    text: str
    keywords: List[str]
    questions: List[str]
    ai_faqs: List[Dict[str, str]]  # {question, answer}
    meta: Dict[str, str]            # {title, description, keywords}
    headings: Dict[str, List[str]]  # {h1: [..], h2: [..]}
    inner_links: List[Dict[str, str]]  # {source_url, anchor_text, target_url, reason}

# -------------------------
# UI Helpers
# -------------------------
def badge(text: str):
    st.markdown(
        f"""
        <span style="background:#eef2ff;color:#4338ca;padding:4px 10px;border-radius:999px;font-size:12px;border:1px solid #c7d2fe;">
        {html.escape(text)}
        </span>
        """,
        unsafe_allow_html=True,
    )

def section_title(title: str, subtitle: Optional[str] = None):
    st.markdown(f"<h2 style='margin-bottom:0'>{html.escape(title)}</h2>", unsafe_allow_html=True)
    if subtitle:
        st.markdown(
            f"<div style='color:#6b7280;margin-top:2px'>{html.escape(subtitle)}</div>",
            unsafe_allow_html=True,
        )

# -------------------------
# Core: Fetch & Parse
# -------------------------
def fetch_page(url: str, timeout: int = 20) -> Tuple[str, str]:
    """Return (title, visible_text) for a URL."""
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36 RivalLensMini/1.0"
            )
        }
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        page_title = soup.title.text.strip() if soup.title else url
        # Remove scripts/styles/nav/footer
        for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
            tag.decompose()
        # Extract visible text
        text = " ".join(t.get_text(separator=" ", strip=True) for t in soup.find_all())
        text = re.sub(r"\s+", " ", text)
        return page_title[:200], text
    except Exception:
        return url, ""

# -------------------------
# Keyword Extraction (simple n-gram frequency)
# -------------------------
def extract_keywords_basic(text: str, top_k: int = 12) -> List[str]:
    text = text.lower()
    tokens = re.findall(r"[a-zA-Z][a-zA-Z\-]+", text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]

    grams: List[str] = []
    for n in [1, 2, 3]:
        for i in range(len(tokens) - n + 1):
            grams.append(" ".join(tokens[i : i + n]))

    freq: Dict[str, int] = {}
    for g in grams:
        freq[g] = freq.get(g, 0) + 1

    scored = [(k, v * (1 + 0.5 * (len(k.split()) - 1))) for k, v in freq.items()]
    scored.sort(key=lambda x: x[1], reverse=True)

    selected: List[str] = []
    for k, _ in scored:
        if any(k in s or s in k for s in selected):
            continue
        selected.append(k)
        if len(selected) >= top_k:
            break
    return selected

# -------------------------
# User Questions via SerpAPI (Google) — optional but robust
# -------------------------
def search_questions_serpapi(query: str, serpapi_key: str, engine: str = "google", num: int = 10) -> List[Dict]:
    """Search with site:reddit or site:quora and return results."""
    if not serpapi_key:
        return []
    try:
        url = "https://serpapi.com/search.json"
        params = {
            "engine": engine,
            "q": query,
            "num": num,
            "api_key": serpapi_key,
            "hl": "en",
            "safe": "active",
        }
        r = requests.get(url, params=params, timeout=30)
        return r.json().get("organic_results", []) if r.status_code == 200 else []
    except Exception:
        return []

def collect_user_questions(keywords: List[str], serpapi_key: str, per_source: int = 5) -> List[str]:
    """Find authentic questions from Reddit/Quora searches."""
    questions: List[str] = []
    if not keywords:
        return questions

    base_queries = [
        q
        for src in USER_QUESTION_SOURCES
        for kw in keywords[:3]
        for q in [
            f"site:{src} {kw} what",
            f"site:{src} {kw} how",
            f"site:{src} {kw} best",
            f"site:{src} {kw} vs",
        ]
    ]

    for q in base_queries:
        results = search_questions_serpapi(q, serpapi_key)
        for res in results:
            title = res.get("title") or ""
            if not title:
                continue
            title = html.unescape(title)
            if title.endswith("?") or re.match(r"^(what|how|why|when|which|where|can|does|do)\b", title.strip().lower()):
                if title not in questions:
                    questions.append(title)
        if len(questions) >= per_source * len(USER_QUESTION_SOURCES):
            break

    uniq: List[str] = []
    for q in questions:
        if q not in uniq:
            uniq.append(q)
    return uniq[: per_source * len(USER_QUESTION_SOURCES)]

# -------------------------
# OpenAI — text generation
# -------------------------
def generate_with_openai(prompt: str, api_key: str, model: str = "gpt-4o-mini") -> str:
    """Lightweight OpenAI call using REST (no sdk dependency)."""
    if not api_key:
        return ""  # handled by demo fallback
    try:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
        }
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=60)
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def demo_generate(prompt: str) -> str:
    return textwrap.shorten(
        (
            "[DEMO OUTPUT] "
            "This is placeholder text because no API key was provided. "
            "Replace with real OpenAI/Gemini output by adding your key in the sidebar.\n\n"
            f"Prompt: {prompt[:300]}"
        ),
        width=1000,
        placeholder="...",
    )

# -------------------------
# Internal Link Recommendations (TF-IDF Similarity)
# -------------------------
def recommend_internal_links(pages: List[PageData], top_n: int = 3) -> List[Dict[str, str]]:
    """For each page, suggest top_n links to other pages based on content similarity."""
    if len(pages) < 2:
        return []

    docs = [p.text for p in pages]
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english")
    X = tfidf.fit_transform(docs)
    sim = cosine_similarity(X)

    recs: List[Dict[str, str]] = []
    for i, _ in enumerate(pages):
        sims = list(enumerate(sim[i]))
        sims = [s for s in sims if s[0] != i]
        sims.sort(key=lambda x: x[1], reverse=True)
        for j, score in sims[:top_n]:
            tgt = pages[j]
            anchor = (pages[i].keywords[:1] or [tgt.title.split("|")[0][:40]])[0]
            recs.append(
                {
                    "source_url": pages[i].url,
                    "anchor_text": anchor,
                    "target_url": tgt.url,
                    "reason": f"High topical similarity ({score:.2f}).",
                }
            )
    return recs

# -------------------------
# AI Orchestration per Page
# -------------------------
def build_ai_prompts(page: PageData) -> Dict[str, str]:
    kw_line = ", ".join(page.keywords[:8])
    faq_block = "\n".join([f"- {q}" for q in page.questions[:DEFAULT_FAQ_COUNT]])

    prompts = {
        "faqs": f"""
You are an SEO & editorial expert. Given the page title and topic keywords below, write crisp, helpful answers (80-150 words) to the FAQs.

Page Title: {page.title}
Topic Keywords: {kw_line}
FAQs:
{faq_block}

Return as a JSON array of objects with keys: question, answer.
""".strip(),
        "meta": f"""
You are an SEO expert. Create meta title (<=60 chars), meta description (<=155 chars) and a comma-separated meta keywords string for the following page.

Page Title: {page.title}
Top Keywords: {kw_line}

Return JSON with keys: title, description, keywords.
""".strip(),
        "headings": f"""
Draft a clean H1 and 6-8 H2s for this topic. Keep H2s actionable and distinct. Return JSON with keys: h1 (string), h2 (array of strings).

Title: {page.title}
Top Keywords: {kw_line}
""".strip(),
    }
    return prompts

def run_ai_generation(page: PageData, api_key: str, model: str, demo_mode: bool = False) -> PageData:
    prompts = build_ai_prompts(page)
    gen = demo_generate if (demo_mode or not api_key) else (lambda p: generate_with_openai(p, api_key, model))

    faqs_raw = gen(prompts["faqs"]) or "[]"
    try:
        page.ai_faqs = json.loads(faqs_raw)
    except Exception:
        page.ai_faqs = [{"question": q, "answer": demo_generate("answer")[:200]} for q in page.questions[:DEFAULT_FAQ_COUNT]]

    meta_raw = gen(prompts["meta"]) or "{}"
    try:
        page.meta = json.loads(meta_raw)
    except Exception:
        page.meta = {
            "title": page.title[:58],
            "description": demo_generate("meta description")[:150],
            "keywords": ", ".join(page.keywords[:8]),
        }

    head_raw = gen(prompts["headings"]) or "{}"
    try:
        j = json.loads(head_raw)
        page.headings = {"h1": [j.get("h1") or page.title], "h2": j.get("h2") or []}
    except Exception:
        page.headings = {"h1": [page.title], "h2": [k.title() for k in page.keywords[:6]]}

    return page

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="🧭", layout="wide")

    # Top Hero
    st.markdown(
        f"""
        <div style="background:linear-gradient(135deg,#f8fafc,#eef2ff);padding:28px;border-radius:24px;border:1px solid #e5e7eb;margin-bottom:14px;">
            <div style="font-size:28px;font-weight:700;">{html.escape(APP_TITLE)}</div>
            <div style="color:#6b7280;margin-top:6px">{html.escape(APP_SUBTITLE)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- SIDEBAR: settings only ---
    with st.sidebar:
        st.header("⚙️ Settings")
        badge("Tip: You can run in demo mode without keys")

        openai_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=os.getenv("OPENAI_API_KEY", ""),
        )

        serpapi_key = st.text_input(
            "SerpAPI Key",
            type="password",
            value=os.getenv("SERPAPI_KEY", ""),
        )

        model = st.selectbox(
            "OpenAI Model",
            ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"],
            index=0,
        )

        faq_count = st.slider("# of FAQs", min_value=3, max_value=8, value=DEFAULT_FAQ_COUNT)
        st.caption("User questions are gathered from Reddit/Quora via Google (SerpAPI)")
        demo_mode = st.toggle("Demo mode (no external calls)", value=not bool(openai_key))
        st.divider()
        st.caption("RivalLens Mini · v1.0")

    # --- MAIN PAGE: Input ---
    section_title("1) Input URLs", "Add up to 10 URLs — we'll fetch, extract keywords & more")
    col1, col2 = st.columns([2, 1])
    with col1:
        urls_text = st.text_area(
            "Paste 1–10 URLs (one per line)",
            height=120,
            placeholder="https://example.com/blog/seo-guide\nhttps://example.com/blog/keyword-research",
        )
    with col2:
        st.write("")
        st.write("")
        st.write("")
        fetch_btn = st.button("Run Automation 🚀", type="primary", use_container_width=True)
        st.caption("We'll crawl pages, pull keywords, find user questions, and generate outputs")

    if fetch_btn:
        raw_urls = [u.strip() for u in urls_text.splitlines() if u.strip()]
        # Unique + limit
        seen = set()
        urls: List[str] = []
        for u in raw_urls:
            if u not in seen:
                seen.add(u)
                urls.append(u)
            if len(urls) >= MAX_URLS:
                break

        if not urls:
            st.warning("Please add at least 1 URL.")
            st.stop()

        st.success(f"Processing {len(urls)} URL(s)...")

        pages: List[PageData] = []
        progress = st.progress(0)
        status = st.empty()

        # Step: fetch & keywords & user questions & AI
        for idx, url in enumerate(urls):
            status.info(f"Fetching: {url}")
            title, text = fetch_page(url)

            if not text:
                st.error(f"Could not fetch or parse content: {url}")
                text = ""

            kw = extract_keywords_basic(text, top_k=14)
            status.info(f"Extracted keywords for {url}")

            q = collect_user_questions(kw, serpapi_key, per_source=faq_count)
            status.info(f"Found {len(q)} authentic user questions for {url}")

            page = PageData(
                url=url,
                title=title,
                text=text,
                keywords=kw,
                questions=q[:faq_count],
                ai_faqs=[],
                meta={},
                headings={},
                inner_links=[],
            )

            status.info(f"Generating AI outputs for {url}")
            page = run_ai_generation(page, openai_key, model, demo_mode=demo_mode)
            pages.append(page)

            progress.progress(int(((idx + 1) / len(urls)) * 100))
            time.sleep(0.1)

        # Step: Internal links (across pages)
        status.info("Calculating internal link recommendations…")
        cross_links = recommend_internal_links(pages, top_n=3)
        for p in pages:
            p.inner_links = [rec for rec in cross_links if rec["source_url"] == p.url]

        status.success("Done! Review results below.")
        st.divider()

        #  Results Tabs per URL
        for p in pages:
            with st.container(border=True):
                st.markdown(f"### 🔗 {p.title}")
                st.caption(p.url)
                chips = " ".join(
                    [
                        f"<span style='background:#f1f5f9;padding:4px 8px;border-radius:999px;border:1px solid #e2e8f0;font-size:12px'>{html.escape(k)}</span>"
                        for k in p.keywords[:10]
                    ]
                )
                st.markdown(chips, unsafe_allow_html=True)

                t1, t2, t3, t4 = st.tabs(["FAQs", "Meta & Headings", "Internal Links", "Raw Content"])

                with t1:
                    if p.questions:
                        st.markdown("**Top user questions (from Reddit/Quora searches):**")
                        for q_text in p.questions:
                            st.markdown(f"- {q_text}")
                    else:
                        st.info("No questions found. Try adding a SerpAPI key or adjust keywords.")

                    st.markdown("**AI Answers:**")
                    if not p.ai_faqs:
                        st.write("No AI output (demo mode).")
                    else:
                        for item in p.ai_faqs[:faq_count]:
                            st.markdown(f"**Q:** {item.get('question','')}\n\n**A:** {item.get('answer','')}")

                with t2:
                    meta = p.meta or {}
                    st.markdown("#### Meta")
                    st.write({k: meta.get(k, "") for k in ["title", "description", "keywords"]})

                    st.markdown("#### Headings")
                    if p.headings:
                        st.write({"H1": p.headings.get("h1", []), "H2": p.headings.get("h2", [])})
                    else:
                        st.info("No headings generated (demo mode).")

                with t3:
                    if p.inner_links:
                        df = pd.DataFrame(p.inner_links)
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("Not enough pages for internal links.")

                with t4:
                    with st.expander("Show extracted plain text"):
                        st.write(p.text[:5000] + ("…" if len(p.text) > 5000 else ""))

        # Export Combined Results
        st.divider()
        section_title("Export Results")
        all_rows: List[Dict[str, str]] = []
        for p in pages:
            faq_flat = json.dumps(p.ai_faqs, ensure_ascii=False)
            h1 = (p.headings.get("h1", [""]) or [""])[0]
            h2 = p.headings.get("h2", [])
            recs = p.inner_links
            all_rows.append(
                {
                    "url": p.url,
                    "page_title": p.title,
                    "top_keywords": ", ".join(p.keywords[:10]),
                    "user_questions": "; ".join(p.questions[:faq_count]),
                    "ai_faqs": faq_flat,
                    "meta_title": p.meta.get("title", ""),
                    "meta_description": p.meta.get("description", ""),
                    "meta_keywords": p.meta.get("keywords", ""),
                    "h1": h1,
                    "h2": "; ".join(h2),
                    "internal_links": json.dumps(recs, ensure_ascii=False),
                }
            )
        export_df = pd.DataFrame(all_rows)
        st.dataframe(export_df, use_container_width=True)

        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        json_bytes = json.dumps(all_rows, ensure_ascii=False, indent=2).encode("utf-8")

        colA, colB = st.columns(2)
        with colA:
            st.download_button("Download CSV", data=csv_bytes, file_name="rivallens_mini_results.csv", mime="text/csv")
        with colB:
            st.download_button("Download JSON", data=json_bytes, file_name="rivallens_mini_results.json", mime="application/json")

    # Footer
    st.markdown(
        """
        <hr/>
        <div style="color:#6b7280;font-size:12px">Tip: Add API keys in the sidebar for live AI and question sourcing. Without keys, app runs in demo mode.</div>
        """,
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
