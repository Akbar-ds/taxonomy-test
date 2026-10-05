from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
import json
import re
import time
from groq import Groq
import pandas as pd
from streamlit_supabase_auth import login_form
from supabase import Client, create_client
import streamlit as st

# ==========================================
# 1. PAGE CONFIG (Must be first Streamlit call)
# ==========================================
st.set_page_config(
    page_title="Master Taxonomy & AI Batch Classifier",
    page_icon="🌍",
    layout="wide",
)

# ==========================================
# 2. CUSTOM CSS STYLING & DATAFRAME FILTER THEME
# ==========================================
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f0f4f8 0%, #f8fafc 100%);
        font-family: 'Inter', sans-serif;
    }
    h1 {
        color: #0f172a;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    div[data-testid="stVerticalBlock"] > div[style*="border"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.03), 0 4px 6px -2px rgba(0, 0, 0, 0.02);
        transition: all 0.3s ease;
    }
    .stButton>button {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        border-radius: 10px;
        font-weight: 600;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.25s ease-in-out;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.2);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(29, 78, 216, 0.3);
        background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%);
    }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# 3. INITIALIZATIONS & CACHED RESOURCES
# ==========================================
@st.cache_resource
def get_groq_client():
    return Groq(api_key=st.secrets["groq"]["api_key"])


@st.cache_resource
def init_connection() -> Client:
    url = st.secrets["supabase"]["url"]
    key = st.secrets["supabase"]["key"]
    return create_client(url, key)


supabase = init_connection()

ALLOWED_TONALITY_OPTIONS = [
    "All Tonalities",
    "Only Positive",
    "Only Negative",
    "Only Neutral",
    "Neutral & Negative",
]


def fetch_master_data():
  try:
    response = (
        supabase.table("taxonomy_entries")
        .select("*")
        .order("created_at", desc=True)
        .execute()
    )
    return response.data
  except Exception as e:
    st.error(f"Error fetching data from Supabase: {e}")
    return []


data = fetch_master_data()
categories = sorted(
    list(set([item.get("category") for item in data if item.get("category")]))
)
subcategories = sorted(
    list(
        set(
            [
                item.get("subcategory")
                for item in data
                if item.get("subcategory")
            ]
        )
    )
)
topics = sorted(
    list(set([item.get("topic") for item in data if item.get("topic")]))
)


# ==========================================
# 4. UTILITY & TEXT CLEANING FUNCTIONS
# ==========================================
def extract_main_topic(topic_str):
  """Strips leading NH numbers, Expressway names, or route designations from topics.

  Example: 'NH-24 Road damage' -> 'Road damage'
  """
  if not topic_str or pd.isna(topic_str):
    return ""
  text = str(topic_str).strip()
  cleaned = re.sub(
      r"^(?:NH-?[0-9A-Za-z]+|[A-Za-z]+(?:-[A-Za-z]+)*\s+(?:EW|Expressway|NH|HWY))\s*[-:]?\s*",
      "",
      text,
      flags=re.IGNORECASE,
  )
  if cleaned == text and "-" in text:
    parts = text.split("-", 1)
    if len(parts) > 1 and any(
        kw in parts[0].lower()
        for kw in ["nh", "expressway", "ew", "hwy", "delhi", "mumbai", "corridor"]
    ):
      cleaned = parts[1].strip()
  return cleaned.strip() if cleaned else text


# ==========================================
# 5. AI BATCH CLASSIFICATION
# ==========================================


def normalize_text(value):
    """Normalize text for reliable comparisons."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_subcategory(value):
    text = normalize_text(value)
    return "" if text in {"", "none", "nan", "null"} else text


def build_supabase_taxonomy(data):
    """
    Build the AI taxonomy exclusively from Supabase.
    Supabase is the single source of truth.
    """
    taxonomy = []

    for tax_id, entry in enumerate(data):
        category = str(entry.get("category", "")).strip()
        topic = str(entry.get("topic", "")).strip()
        subcategory_raw = entry.get("subcategory")
        tonality_rule = str(
            entry.get("tonality", "All Tonalities")
        ).strip()

        if not category or not topic:
            continue

        if (
            subcategory_raw is None
            or pd.isna(subcategory_raw)
            or str(subcategory_raw).strip().lower()
            in {"", "none", "nan", "null"}
        ):
            subcategory = ""
        else:
            subcategory = str(subcategory_raw).strip()

        taxonomy.append({
            "Taxonomy ID": tax_id,
            "Category": category,
            "Subcategory": subcategory,
            "Topic": topic,
            "Tonality Rule": tonality_rule,
        })

    return taxonomy


CLASSIFIER_STOP_WORDS = {
    "the", "a", "an", "and", "or", "but", "of", "to", "in", "on",
    "for", "from", "with", "by", "at", "is", "are", "was", "were",
    "has", "have", "had", "this", "that", "these", "those", "as", "it",
    "its", "be", "will", "would", "could", "should", "into", "over",
    "under", "after", "before", "about", "than", "also", "said", "says",
    "their", "there", "they", "them", "he", "she", "his", "her", "we",
    "our", "you", "your", "which", "who", "what", "when", "where", "why",
    "how", "while", "during", "through", "new", "one", "two", "three",
}


def _tokens(text):
    return {
        word
        for word in re.findall(r"[a-z0-9]+", normalize_text(text))
        if len(word) >= 3 and word not in CLASSIFIER_STOP_WORDS
    }


def _phrases(text):
    tokens = re.findall(r"[a-z0-9]+", normalize_text(text))
    return {
        " ".join(tokens[i:i + 2])
        for i in range(len(tokens) - 1)
    } | {
        " ".join(tokens[i:i + 3])
        for i in range(len(tokens) - 2)
    }


def _best_phrase_similarity(label, article_sentences):
    """Approximate semantic/wording similarity without another API call."""
    label_tokens = _tokens(label)
    if not label_tokens or not article_sentences:
        return 0.0

    best = 0.0
    for sentence in article_sentences[:30]:
        sentence_tokens = _tokens(sentence)
        if not sentence_tokens:
            continue

        overlap = len(label_tokens & sentence_tokens) / max(
            1, len(label_tokens)
        )
        fuzzy = SequenceMatcher(
            None,
            normalize_text(label),
            normalize_text(sentence)[:500],
        ).ratio()
        best = max(best, (overlap * 0.8) + (fuzzy * 0.2))

    return best


def taxonomy_matches_article(article_text, taxonomy, max_candidates=18):
    """
    Rank Supabase taxonomy records for an article.

    Important: never append arbitrary database rows just to fill the
    candidate list. That was one of the weaknesses of the previous version.
    """
    article = normalize_text(article_text)
    article_words = _tokens(article)
    article_phrases = _phrases(article)
    sentences = [
        s.strip()
        for s in re.split(r"[.!?;\n]+", article)
        if s.strip()
    ]

    scored = []

    for item in taxonomy:
        topic = normalize_text(item["Topic"])
        category = normalize_text(item["Category"])
        subcategory = normalize_text(item["Subcategory"])

        topic_words = _tokens(topic)
        sub_words = _tokens(subcategory)
        cat_words = _tokens(category)

        score = 0.0

        # Topic is the strongest signal.
        topic_overlap = len(article_words & topic_words)
        score += topic_overlap * 7.0

        # Subcategory/category help route related topics.
        score += len(article_words & sub_words) * 2.5
        score += len(article_words & cat_words) * 1.5

        # Exact phrase matches are very strong.
        if topic and topic in article:
            score += 30.0
        if subcategory and subcategory in article:
            score += 8.0
        if category and category in article:
            score += 4.0

        # Bigram/trigram phrase overlap.
        label_phrases = _phrases(topic)
        score += len(article_phrases & label_phrases) * 4.0

        # Approximate wording similarity against article sentences.
        score += _best_phrase_similarity(topic, sentences) * 6.0

        scored.append((score, item))

    scored.sort(key=lambda pair: pair[0], reverse=True)

    # If the taxonomy is small, give Qwen the complete taxonomy.
    if len(scored) <= max_candidates:
        return [item for _, item in scored]

    return [item for _, item in scored[:max_candidates]]


def allowed_tonalities(rule):
    rule = str(rule or "All Tonalities").strip()

    if rule == "Only Positive":
        return ["Positive"]
    if rule == "Only Negative":
        return ["Negative"]
    if rule == "Only Neutral":
        return ["Neutral"]
    if rule == "Neutral & Negative":
        return ["Neutral", "Negative"]
    return ["Positive", "Negative", "Neutral"]


def canonicalize_result(result, full_taxonomy, candidate_taxonomy):
    """
    Hard validation.

    Qwen returns only a Taxonomy ID. Python then retrieves the exact
    Category/Subcategory/Topic/Tonality Rule from Supabase.
    """
    if not isinstance(result, dict):
        return None

    try:
        tax_id = int(result.get("Taxonomy ID"))
    except (TypeError, ValueError):
        return None

    candidate_ids = {
        int(item["Taxonomy ID"])
        for item in candidate_taxonomy
    }

    if tax_id not in candidate_ids:
        return None

    record = next(
        (
            item for item in full_taxonomy
            if int(item["Taxonomy ID"]) == tax_id
        ),
        None,
    )

    if record is None:
        return None

    rule = record.get("Tonality Rule", "All Tonalities")
    allowed = allowed_tonalities(rule)

    ai_tonality = str(
        result.get("Tonality", "")
    ).strip().capitalize()

    if ai_tonality not in allowed:
        # For fixed database rules, Python can safely enforce the rule.
        # For All Tonalities, the AI must provide a valid sentiment.
        if len(allowed) == 1:
            ai_tonality = allowed[0]
        else:
            return None

    # Overall Tonality is intentionally canonicalized to the validated
    # Tonality so the two fields can never conflict.
    return {
        "index": result.get("index"),
        "Category": record["Category"],
        "Subcategory": record["Subcategory"],
        "Topic": record["Topic"],
        "Tonality": ai_tonality,
        "Overall Tonality": ai_tonality,
        "NH NO": str(result.get("NH NO", "N/A")).strip() or "N/A",
        "Location": str(result.get("Location", "N/A")).strip() or "N/A",
    }


def _is_rate_or_size_error(error):
    text = str(error).lower()
    return any(
        marker in text
        for marker in (
            "rate_limit",
            "rate limit",
            "429",
            "413",
            "tokens per minute",
            "request too large",
        )
    )


def _groq_json_call(client, prompt, max_completion_tokens=1200):
    """Single deterministic Groq JSON request."""
    return client.chat.completions.create(
        model="qwen/qwen3.6-27b",
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0,
        reasoning_effort="none",
        reasoning_format="hidden",
        response_format={"type": "json_object"},
        max_completion_tokens=max_completion_tokens,
    )


def _build_classification_prompt(articles_payload, candidate_taxonomy):
    article_blocks = []

    for article in articles_payload:
        idx = article["index"]
        candidates = candidate_taxonomy[idx]

        compact_candidates = [
            {
                "Taxonomy ID": item["Taxonomy ID"],
                "Category": item["Category"],
                "Subcategory": item["Subcategory"],
                "Topic": item["Topic"],
                "Tonality Rule": item["Tonality Rule"],
            }
            for item in candidates
        ]

        article_blocks.append(
            "ARTICLE INDEX: " + str(idx) + "\n"
            "ARTICLE:\n" + str(article.get("content", "")) + "\n"
            "VALID TAXONOMY RECORDS:\n" +
            json.dumps(
                compact_candidates,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    return f"""
You are the MoRTH media classification engine.

Classify every article using ONLY the supplied taxonomy records.

HARD RULES:
1. Return exactly one result for every article.
2. Choose exactly ONE Taxonomy ID from that article's VALID TAXONOMY RECORDS.
3. Never invent a Taxonomy ID.
4. Never invent, rename, shorten or paraphrase a Topic.
5. Category, Subcategory and Topic are taken from the selected database record.
6. Never use Miscellaneous, Other, General or Unknown unless that exact Topic is explicitly present in the supplied records.
7. Overall Tonality MUST equal Tonality.
8. Obey the selected record's Tonality Rule exactly.
9. Always Choose the tonality based on the sentiment of the content towards the client MoRTH or its entities NHAI, NHIDCL, BRO, Nitin Gadkari, NRSC
10.If the content can be classified in multiple topics always choose the one which is more dominant throughout the content the core issue of the article to be considered for taking topics.

TONALITY:
- Always Choose the tonality based on the sentiment of the content towards the client MoRTH or its entities NHAI, NHIDCL, BRO, Nitin Gadkari, NRSC
- Positive: the article positively affects public perception of MoRTH/NHAI/NHIDCL/BRO,
  Nitin Gadkari in his ministerial/client role, or a MoRTH-associated project, policy,
  scheme, authority or infrastructure asset.
- Negative: criticism, allegations, corruption, delays, failures, negligence,
  accidents attributed to client action/inaction, protests, disputes, safety failures,
  poor execution or other reporting that harms the client's public perception.
- Neutral: factual reporting without a meaningful positive or negative client impact.
- Ignore sentiment toward unrelated entities unless it directly affects the client.

DATABASE TONALITY RULES:
- Only Positive -> Positive only.
- Only Negative -> Negative only.
- Only Neutral -> Neutral only.
- Neutral & Negative -> Neutral or Negative only.
- All Tonalities -> choose Positive, Negative or Neutral from the article.

TOPIC:
If the content can be classified in multiple topics always choose the one which is more dominant throughout the content the core issue of the article to be considered for taking topics.
Choose the topic that best represents the article's MAIN subject, not a minor mention.
Prefer the most specific applicable topic over a broad one.
Do not choose a topic merely because one word happens to appear in the article.

NH NO:
Return the highway number/name only when supported by the article. Otherwise N/A.

LOCATION:
Return the main location and state when supported by the article. Otherwise N/A.

OUTPUT JSON ONLY:
{{
  "results": [
    {{
      "index": 0,
      "Taxonomy ID": 0,
      "Tonality": "Positive",
      "NH NO": "N/A",
      "Location": "N/A"
    }}
  ]
}}

ARTICLES:

{chr(10).join(article_blocks)}
"""


def _build_repair_prompt(invalid_articles, candidate_taxonomy):
    blocks = []

    for article in invalid_articles:
        idx = article["index"]
        candidates = [
            {
                "Taxonomy ID": item["Taxonomy ID"],
                "Category": item["Category"],
                "Subcategory": item["Subcategory"],
                "Topic": item["Topic"],
                "Tonality Rule": item["Tonality Rule"],
            }
            for item in candidate_taxonomy[idx]
        ]

        blocks.append(
            f"ARTICLE INDEX: {idx}\n"
            f"ARTICLE:\n{article.get('content', '')}\n"
            "VALID TAXONOMY RECORDS:\n"
            + json.dumps(
                candidates,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

    return f"""
Correct the classification for the supplied articles.

The previous answer was invalid because it did not satisfy the taxonomy rules.

For each article:
- Choose exactly one Taxonomy ID from its supplied records.
- Do not invent a Topic.
- Do not output Miscellaneous unless it is explicitly supplied.
- Category/Subcategory/Topic come from the selected record.
- Obey the selected record's Tonality Rule.
- Overall Tonality must equal Tonality.
- Return JSON only.

{{
  "results": [
    {{
      "index": 0,
      "Taxonomy ID": 0,
      "Tonality": "Neutral",
      "NH NO": "N/A",
      "Location": "N/A"
    }}
  ]
}}

{chr(10).join(blocks)}
"""


@st.cache_data(show_spinner=False)
def classify_batch_articles(articles_json_str, taxonomy_reference):
    """Classify a batch with Groq/Qwen and hard-validate against Supabase."""
    client = get_groq_client()
    articles_payload = json.loads(articles_json_str)
    full_taxonomy = json.loads(taxonomy_reference)

    if not full_taxonomy:
        return [
            {
                "index": item["index"],
                "Category": "Error",
                "Subcategory": "No taxonomy",
                "Topic": "N/A",
                "Tonality": "N/A",
                "Overall Tonality": "N/A",
                "NH NO": "N/A",
                "Location": "N/A",
            }
            for item in articles_payload
        ]

    candidate_taxonomy = {
        article["index"]: taxonomy_matches_article(
            article.get("content", ""),
            full_taxonomy,
            max_candidates=18,
        )
        for article in articles_payload
    }

    # If a candidate list somehow becomes empty, use the complete taxonomy only
    # for that article. This is rare and avoids an invalid/empty AI request.
    for article in articles_payload:
        if not candidate_taxonomy[article["index"]]:
            candidate_taxonomy[article["index"]] = full_taxonomy[:24]

    prompt = _build_classification_prompt(
        articles_payload,
        candidate_taxonomy,
    )

    last_error = None

    for attempt in range(4):
        try:
            response = _groq_json_call(
                client,
                prompt,
                max_completion_tokens=1200,
            )

            raw = response.choices[0].message.content
            parsed = json.loads(raw)
            ai_results = parsed.get("results", []) if isinstance(parsed, dict) else []

            by_index = {
                str(result.get("index")): result
                for result in ai_results
                if isinstance(result, dict)
            }

            validated = {}
            invalid_articles = []

            for article in articles_payload:
                idx = article["index"]
                result = by_index.get(str(idx))

                canonical = canonicalize_result(
                    result,
                    full_taxonomy,
                    candidate_taxonomy[idx],
                )

                if canonical is None:
                    invalid_articles.append(article)
                else:
                    validated[idx] = canonical

            # Everything passed on the first attempt.
            if not invalid_articles:
                return [validated[a["index"]] for a in articles_payload]

            # One targeted repair call for only invalid articles.
            repair_prompt = _build_repair_prompt(
                invalid_articles,
                candidate_taxonomy,
            )

            repair_response = _groq_json_call(
                client,
                repair_prompt,
                max_completion_tokens=900,
            )

            repair_raw = repair_response.choices[0].message.content
            repair_parsed = json.loads(repair_raw)
            repair_results = (
                repair_parsed.get("results", [])
                if isinstance(repair_parsed, dict)
                else []
            )

            repair_by_index = {
                str(result.get("index")): result
                for result in repair_results
                if isinstance(result, dict)
            }

            for article in invalid_articles:
                idx = article["index"]
                repaired = canonicalize_result(
                    repair_by_index.get(str(idx)),
                    full_taxonomy,
                    candidate_taxonomy[idx],
                )
                if repaired is not None:
                    validated[idx] = repaired

            # If repair still failed, do NOT invent a taxonomy record.
            # Mark the row clearly for review rather than silently producing
            # a wrong topic.
            final_results = []
            for article in articles_payload:
                idx = article["index"]
                if idx in validated:
                    final_results.append(validated[idx])
                else:
                    final_results.append({
                        "index": idx,
                        "Category": "REVIEW REQUIRED",
                        "Subcategory": "REVIEW REQUIRED",
                        "Topic": "REVIEW REQUIRED",
                        "Tonality": "REVIEW REQUIRED",
                        "Overall Tonality": "REVIEW REQUIRED",
                        "NH NO": "N/A",
                        "Location": "N/A",
                    })

            return final_results

        except Exception as error:
            last_error = error

            error_text = str(error).lower()

            # A 413/request-too-large error will not be fixed by waiting.
            # Return a specific marker so classify_in_batches can split the
            # batch automatically into smaller requests.
            if (
                "413" in error_text
                or "request too large" in error_text
                or "tokens per minute" in error_text
            ):
                return [
                    {
                        "index": item["index"],
                        "Category": "Error",
                        "Subcategory": "REQUEST_TOO_LARGE: " + str(error),
                        "Topic": "N/A",
                        "Tonality": "N/A",
                        "Overall Tonality": "N/A",
                        "NH NO": "N/A",
                        "Location": "N/A",
                    }
                    for item in articles_payload
                ]

            if _is_rate_or_size_error(error) and attempt < 3:
                wait_time = 4 * (2 ** attempt)
                time.sleep(wait_time)
                continue

            if attempt < 3:
                time.sleep(2 ** attempt)
                continue

    return [
        {
            "index": item["index"],
            "Category": "Error",
            "Subcategory": str(last_error),
            "Topic": "N/A",
            "Tonality": "N/A",
            "Overall Tonality": "N/A",
            "NH NO": "N/A",
            "Location": "N/A",
        }
        for item in articles_payload
    ]


def classify_in_batches(df_articles, taxonomy_data, batch_size):
    """Process articles in controlled batches and auto-split oversized requests."""
    taxonomy = build_supabase_taxonomy(taxonomy_data)
    taxonomy_reference = json.dumps(
        taxonomy,
        ensure_ascii=False,
        separators=(",", ":"),
    )

    if not taxonomy:
        raise ValueError(
            "No taxonomy records were found in Supabase taxonomy_entries."
        )

    results_by_index = {}
    progress_bar = st.progress(0)
    total_rows = len(df_articles)
    processed_count = 0

    def process_batch(batch_df):
        nonlocal processed_count

        if batch_df.empty:
            return

        articles_payload = [
            {
                "index": int(idx),
                "content": str(row.get("content_snippet", "")),
            }
            for idx, row in batch_df.iterrows()
        ]

        parsed_batch = classify_batch_articles(
            json.dumps(
                articles_payload,
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            taxonomy_reference,
        )

        oversized = any(
            str(item.get("Subcategory", "")).startswith(
                "REQUEST_TOO_LARGE:"
            )
            for item in parsed_batch
        )

        if oversized and len(batch_df) > 1:
            midpoint = max(1, len(batch_df) // 2)
            process_batch(batch_df.iloc[:midpoint])
            process_batch(batch_df.iloc[midpoint:])
            return

        for item in parsed_batch:
            results_by_index[item["index"]] = item

        processed_count += len(batch_df)
        progress_bar.progress(
            min(processed_count / total_rows, 1.0)
        )

    for start in range(0, total_rows, batch_size):
        process_batch(
            df_articles.iloc[start:start + batch_size]
        )

    # Preserve exact input order and guarantee one result row per article.
    result_rows = []
    for idx in df_articles.index:
        result_rows.append(
            results_by_index.get(
                int(idx),
                {
                    "index": int(idx),
                    "Category": "Error",
                    "Subcategory": "Missing result",
                    "Topic": "N/A",
                    "Tonality": "N/A",
                    "Overall Tonality": "N/A",
                    "NH NO": "N/A",
                    "Location": "N/A",
                },
            )
        )

    result_df = pd.DataFrame(result_rows).drop(columns=["index"])

    return pd.concat(
        [
            df_articles.reset_index(drop=True),
            result_df.reset_index(drop=True),
        ],
        axis=1,
    )


# ==========================================
# 6. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.title("🧭 Explore")
app_mode = st.sidebar.selectbox(
    "Choose Application Mode",
    ["📋 Master Taxonomy Manager", "🤖 MoRTH AI", "🔍 MoRTH QC"],
)


# ==========================================
# 7. MODAL DIALOGS
# ==========================================
@st.dialog("📝 Add New Taxonomy Entry")
def add_taxonomy_modal(categories, subcategories, ALLOWED_TONALITY_OPTIONS):
  with st.form("new_entry_form", clear_on_submit=True):
    op_name = st.text_input(
        "Your Name / User ID *", placeholder="Enter your full name..."
    )
    form_category = st.selectbox(
        "Category *",
        options=categories
        if categories
        else [
            "Interviews",
            "Grievances",
            "Projects and Infra",
            "Policies",
            "Analysis Report",
            "Rules Violation",
            "Technology",
            "Irregularities",
        ],
    )
    form_subcategory = st.selectbox(
        "Subcategory (Optional)", options=[""] + subcategories
    )
    form_topic = st.text_input("Topic *", placeholder="Enter topic name...")
    form_tonality = st.selectbox(
        "Tonality / Condition Rule *", options=ALLOWED_TONALITY_OPTIONS
    )

    submit_col, cancel_col = st.columns(2)
    with submit_col:
      submitted = st.form_submit_button(
          "🚀 Submit & Sync", type="primary", use_container_width=True
      )
    with cancel_col:
      cancelled = st.form_submit_button("❌ Cancel", use_container_width=True)

    if submitted:
      cleaned_topic = form_topic.strip()
      cleaned_name = op_name.strip()
      if not cleaned_name or not cleaned_topic:
        st.error("Please fill out your Name and the Topic field.")
      else:
        try:
          supabase.table("taxonomy_entries").insert({
              "category": form_category,
              "subcategory": form_subcategory
              if form_subcategory != ""
              else None,
              "topic": cleaned_topic,
              "tonality": form_tonality,
              "submitted_by": cleaned_name,
          }).execute()
          st.success("✨ Entry successfully added and synced!")
          st.rerun()
        except Exception as e:
          st.error(f"Failed to add entry: {e}")
    if cancelled:
      st.rerun()


@st.dialog("✏️ Edit Taxonomy Entry")
def edit_taxonomy_modal(
    item, categories, subcategories, ALLOWED_TONALITY_OPTIONS
):
  with st.form("edit_entry_form", clear_on_submit=False):
    editor_name = st.text_input(
        "Your Name / User ID (Editor) *", placeholder="Enter your full name..."
    )
    default_cat_idx = (
        categories.index(item.get("category"))
        if item.get("category") in categories
        else 0
    )
    edit_category = st.selectbox(
        "Category *", options=categories, index=default_cat_idx
    )

    sub_opts = [""] + subcategories
    default_sub_idx = (
        sub_opts.index(item.get("subcategory"))
        if item.get("subcategory") in sub_opts
        else 0
    )
    edit_subcategory = st.selectbox(
        "Subcategory (Optional)", options=sub_opts, index=default_sub_idx
    )

    edit_topic = st.text_input("Topic *", value=item.get("topic", ""))
    default_ton_idx = (
        ALLOWED_TONALITY_OPTIONS.index(item.get("tonality"))
        if item.get("tonality") in ALLOWED_TONALITY_OPTIONS
        else 0
    )
    edit_tonality = st.selectbox(
        "Tonality / Condition Rule *",
        options=ALLOWED_TONALITY_OPTIONS,
        index=default_ton_idx,
    )

    proceed_col, cancel_col = st.columns(2)
    with proceed_col:
      proceed_button = st.form_submit_button(
          "✅ Proceed & Update", type="primary", use_container_width=True
      )
    with cancel_col:
      cancel_button = st.form_submit_button("❌ Cancel", use_container_width=True)

    if proceed_button:
      cleaned_topic = edit_topic.strip()
      cleaned_name = editor_name.strip()
      if not cleaned_name or not cleaned_topic:
        st.error("Please fill out your Name and the Topic field.")
      else:
        try:
          update_payload = {
              "category": edit_category,
              "subcategory": edit_subcategory
              if edit_subcategory != ""
              else None,
              "topic": cleaned_topic,
              "tonality": edit_tonality,
              "submitted_by": cleaned_name,
              "created_at": datetime.utcnow().isoformat(),
          }
          supabase.table("taxonomy_entries").update(update_payload).eq(
              "id", item.get("id")
          ).execute()
          st.success("Edited in the database successfully!")
          st.rerun()
        except Exception as e:
          st.error(f"Failed to update entry in database: {e}")
    if cancel_button:
      st.rerun()


# ==========================================
# 8. EXCEL-LIKE COLUMN FILTERING UTILITY (Matching user image layout)
# ==========================================
def render_excel_style_qc_section(df, display_columns, section_key_prefix):
  """Renders clean Excel-like multi-select filter dropdown boxes directly above each column,

  matching the provided interface layout exactly ("All Selected" dropdown boxes per column).
  """
  working_df = df.copy()
  available_cols = [c for c in display_columns if c in working_df.columns]
  working_df = working_df[available_cols]

  filtered_df = working_df.copy()

  # Render Excel-style column filter controls side-by-side matching headers
  st.markdown("### 🎛️ Column Filters")
  filter_cols = st.columns(len(available_cols))

  for idx, col_name in enumerate(available_cols):
    with filter_cols[idx]:
      unique_vals = sorted(
          [str(v) for v in working_df[col_name].dropna().unique()]
      )
      
      # Use multi-select mimicking Excel filter dropdown field
      selected_vals = st.multiselect(
          label=col_name,
          options=unique_vals,
          default=[],
          placeholder="All Selected",
          key=f"{section_key_prefix}_excel_col_{col_name}",
      )

      if selected_vals:
        filtered_df = filtered_df[
            filtered_df[col_name].astype(str).isin(selected_vals)
        ]

  st.divider()
  st.subheader(f"📊 Filtered Results ({len(filtered_df)} rows)")

  # Action Buttons Row
  btn_col1, btn_col2, _ = st.columns([1, 1, 3])

  with btn_col1:
    id_col_candidates = ["Article ID", "ArticleID", "article_id", "ID"]
    match_id_col = next(
        (c for c in id_col_candidates if c in filtered_df.columns), None
    )

    if match_id_col:
      unique_ids = (
          filtered_df[match_id_col].dropna().astype(str).unique().tolist()
      )
      ids_string = ", ".join(unique_ids)
      if st.button("📋 Copy Unique ID's", key=f"{section_key_prefix}_copy_btn"):
        st.code(ids_string, language="text")
        st.success(f"Copied {len(unique_ids)} unique IDs successfully!")

  with btn_col2:
    if not filtered_df.empty:
      csv_bytes = filtered_df.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Download Dataset",
          data=csv_bytes,
          file_name=f"{section_key_prefix}_report.csv",
          mime="text/csv",
          key=f"{section_key_prefix}_download_btn",
      )

  st.dataframe(filtered_df, use_container_width=True)
  return filtered_df


# ==========================================
# 9. MAIN APP INTERFACE ROUTING
# ==========================================
if app_mode == "🤖 MoRTH AI":
  st.subheader("🤖 HELLO I AM MoRTH AI")
  st.markdown(
      "<p style='font-size: 15px; color: #475569;'>Choose whether to upload a"
      " spreadsheet or add content snippets sequentially below for instant"
      " AI classification.</p>",
      unsafe_allow_html=True,
  )
  st.divider()

  input_method = st.radio(
      "Select Input Source",
      ["📁 Upload Spreadsheet (CSV/Excel)", "✍️ Add Contents Sequentially"],
      horizontal=True,
  )

  df_input = None

  if input_method == "📁 Upload Spreadsheet (CSV/Excel)":
    uploaded_file = st.file_uploader(
        "Upload Excel or CSV containing your articles", type=["xlsx", "csv"]
    )
    if uploaded_file:
      df_input = (
          pd.read_csv(uploaded_file)
          if uploaded_file.name.endswith(".csv")
          else pd.read_excel(uploaded_file)
      )
      st.write("Preview of Uploaded Data:", df_input.head())

  else:
    st.markdown("### ✍️ Sequential Content Input")
    st.markdown(
        "<small style='color: #64748b;'>Paste your content snippet below and"
        " click the button to add it to your queue.</small>",
        unsafe_allow_html=True,
    )

    if "sequential_snippets" not in st.session_state:
      st.session_state.sequential_snippets = []
    if "cached_output_df" not in st.session_state:
      st.session_state.cached_output_df = None

    # Handle text input widget without modifying state directly via assignment
    snippet_text_input = st.text_area(
        "Enter content snippet",
        placeholder="Paste your content snippet here...",
        height=100,
        key="current_snippet_input_widget"
    )

    col_btn1, _ = st.columns([1, 4])
    with col_btn1:
      add_item_btn = st.button("➕ Add Content Item", use_container_width=True)

    if add_item_btn:
      snippet_text = snippet_text_input.strip()
      if snippet_text:
        st.session_state.sequential_snippets.append(snippet_text)
        st.session_state.cached_output_df = None
        st.rerun()
      else:
        st.warning("Please enter some content before adding.")

    if st.session_state.sequential_snippets:
      st.markdown("#### 📋 Added Contents Queue:")
      for idx, snip in enumerate(st.session_state.sequential_snippets):
        st.markdown(
            f"**{idx + 1}.** {snip[:120]}{'...' if len(snip) > 120 else ''}"
        )

      col_reset, _ = st.columns([1, 4])
      with col_reset:
        if st.button("🗑️ Clear All Items"):
          st.session_state.sequential_snippets = []
          st.session_state.cached_output_df = None
          st.rerun()

      df_input = pd.DataFrame(
          {"content_snippet": st.session_state.sequential_snippets}
      )

  if df_input is not None and not df_input.empty:
    batch_size = st.slider(
        "Batch Size (Articles per AI Request)",
        min_value=1,
        max_value=10,
        value=3,
        help="Use 3-5 on the Groq free tier. You can increase this on a higher rate limit.",
    )

    if st.button("🚀 Run AI Classification"):
      if "content_snippet" not in df_input.columns:
        st.error(
            "Error: Data must contain a column named 'content_snippet'."
        )
      elif not data:
        st.error(
            "No taxonomy records were found in Supabase. Please add taxonomy entries first."
        )
      else:
        try:
          with st.spinner("Processing articles through Qwen 3.6 27B..."):
            st.session_state.cached_output_df = classify_in_batches(
                df_input,
                data,
                batch_size,
            )
        except Exception as e:
          st.error(f"Classification failed: {e}")

    if st.session_state.get("cached_output_df") is not None:
      st.success("✨ Classification complete!")
      st.dataframe(st.session_state.cached_output_df)

      csv_data = (
          st.session_state.cached_output_df.to_csv(index=False).encode("utf-8")
      )
      st.download_button(
          label="📥 Download Categorized Results as CSV",
          data=csv_data,
          file_name="categorized_articles_output.csv",
          mime="text/csv",
      )

elif app_mode == "🔍 MoRTH QC":
  st.subheader("🔍 MoRTH Quality Control (QC Dashboard)")
  st.markdown(
      "<p style='font-size: 15px; color: #475569;'>Upload your analysis Excel"
      " report below to run automated validations across specific QC"
      " sub-modules with embedded Excel-style header filters.</p>",
      unsafe_allow_html=True,
  )
  st.divider()

  qc_file = st.file_uploader(
      "Upload Master Analysis Report (Excel/CSV)", type=["xlsx", "csv"]
  )

  if qc_file:
    qc_df = (
        pd.read_csv(qc_file)
        if qc_file.name.endswith(".csv")
        else pd.read_excel(qc_file)
    )

    # Sub-tabs for MoRTH QC
    qc_tabs = st.tabs([
        "👤 Journalist QC",
        "🗂️ Topic & Taxonomy QC",
        "📸 Photo QC",
        "🗣️ Spokes QC",
        "⚖️ Conflicts",
        "⚠️ Blank Tonality QC",
    ])

    # 1. Journalist QC
    with qc_tabs[0]:
      st.markdown("### 👤 Journalist Quality Control")
      cols = ["Medium", "Article ID", "Analysis By", "Analysis By Bureau", "Journalist"]
      render_excel_style_qc_section(qc_df, cols, "journalist_qc")

    # 2. Topic Category and Sub Category QC
    with qc_tabs[1]:
      st.markdown("### 🗂️ Topic, Category & Sub-Category QC")
      st.markdown(
          "<small style='color: #64748b;'>Validating row attributes, leading"
          " expressway route prefixes, subcategory null handling, and database"
          " tonality rule matrices.</small>",
          unsafe_allow_html=True,
      )

      db_rules_map = {}
      for entry in data:
        db_top = extract_main_topic(entry.get("topic", ""))
        db_cat = str(entry.get("category", "")).strip().lower()
        db_sub = entry.get("subcategory")
        if db_sub is None or pd.isna(db_sub) or str(db_sub).strip().lower() in ["none", "nan", ""]:
          db_sub_norm = ""
        else:
          db_sub_norm = str(db_sub).strip().lower()
          
        db_ton_rule = str(entry.get("tonality", "All Tonalities")).strip()
        db_rules_map[(db_top.lower(), db_cat, db_sub_norm)] = db_ton_rule

      validation_rows = []
      for idx, row in qc_df.iterrows():
        raw_top = row.get("Topic", "")
        clean_top = extract_main_topic(raw_top).lower()
        
        r_cat = str(row.get("Category", "")).strip().lower()
        
        r_sub_raw = row.get("Sub Category1", "")
        if r_sub_raw is None or pd.isna(r_sub_raw) or str(r_sub_raw).strip().lower() in ["none", "nan", ""]:
          r_sub = ""
        else:
          r_sub = str(r_sub_raw).strip().lower()

        r_ton = str(row.get("Tonality", "")).strip()

        match_key = (clean_top, r_cat, r_sub)
        
        if match_key in db_rules_map:
          db_rule = db_rules_map[match_key]
          tonality_valid = True
          if db_rule == "Only Positive":
            if r_ton.lower() != "positive":
              tonality_valid = False
          elif db_rule == "Only Negative":
            if r_ton.lower() != "negative":
              tonality_valid = False
          elif db_rule == "Only Neutral":
            if r_ton.lower() != "neutral":
              tonality_valid = False
          elif db_rule == "Neutral & Negative":
            if r_ton.lower() not in ["neutral", "negative"]:
              tonality_valid = False

          if tonality_valid:
            rule_msg = "Correct Match"
          else:
            rule_msg = f"Tonality Rule Mismatch (Expected DB Rule: {db_rule})"
        else:
          matching_topics = [k[0] for k in db_rules_map.keys()]
          matching_cats = [k[1] for k in db_rules_map.keys()]
          
          if clean_top not in matching_topics:
            rule_msg = f"Mismatch: Topic '{extract_main_topic(raw_top)}' not found in DB"
          elif r_cat not in matching_cats:
            rule_msg = f"Mismatch: Category '{row.get('Category')}' does not match topic rules"
          else:
            rule_msg = "Mismatch: Subcategory or rule combination mismatch"

        validation_rows.append(rule_msg)

      val_df = qc_df.copy()
      val_df["Rule Logic"] = validation_rows

      cols_t = [
          "Medium",
          "Article ID",
          "Analysis By",
          "Topic",
          "Category",
          "Sub Category1",
          "Tonality",
          "Analysis By Bureau",
          "Rule Logic",
      ]
      render_excel_style_qc_section(val_df, cols_t, "topic_cat_qc")

    # 3. Photo QC
    with qc_tabs[2]:
      st.markdown("### 📸 Photo Quality Control")
      cols_photo = [
          "Article ID",
          "Medium",
          "Analysis By",
          "Analysis By Bureau",
          "Photo Mention",
          "Topic",
      ]
      render_excel_style_qc_section(qc_df, cols_photo, "photo_qc")

    # 4. Spokes QC
    with qc_tabs[3]:
      st.markdown("### 🗣️ Spokes Quality Control")
      spokes_df = qc_df.copy()
      flags = []
      for idx, row in spokes_df.iterrows():
        spoke_val = str(row.get("Spokes", "")).strip()
        quote_val = str(row.get("Quotes", "")).strip().lower()

        if spoke_val and spoke_val.lower() != "nan" and spoke_val != "":
          if quote_val in ["", "nan", "no", "blank", "none"]:
            flags.append("Missing Quotes")
          elif quote_val in ["yes", "true", "1"]:
            flags.append("Review Entry")
          else:
            flags.append("Missing Quotes")
        else:
          flags.append("OK")

      spokes_df["Flag"] = flags
      cols_spokes = [
          "Article ID",
          "Medium",
          "Analysis By",
          "Analysis By Bureau",
          "Spokes",
          "Quotes",
          "Flag",
      ]
      render_excel_style_qc_section(spokes_df, cols_spokes, "spokes_qc")

    # 5. Conflicts
    with qc_tabs[4]:
      st.markdown("### ⚖️ Tonality Conflicts QC")
      conflict_df = qc_df.copy()
      conflict_flags = []
      filtered_conflict_rows = []

      for idx, row in conflict_df.iterrows():
        overall_ton = str(row.get("Overall Tonality", "")).strip()
        row_ton = str(row.get("Tonality", "")).strip()

        is_missing = (
            not overall_ton
            or overall_ton.lower() in ["nan", "none", ""]
            or overall_ton.isspace()
        )
        is_mismatch = (
            not is_missing
            and row_ton
            and row_ton.lower() not in ["nan", "none", ""]
            and overall_ton.lower() != row_ton.lower()
        )

        if is_missing:
          conflict_flags.append("Missing Tonality")
          filtered_conflict_rows.append(row)
        elif is_mismatch:
          conflict_flags.append("Mismatched")
          filtered_conflict_rows.append(row)

      if filtered_conflict_rows:
        conf_result_df = pd.DataFrame(filtered_conflict_rows)
        conf_result_df["Flag"] = [
            (
                "Missing Tonality"
                if not str(o).strip() or str(o).lower() in ["nan", "none", ""]
                else "Mismatched"
            )
            for o in conf_result_df["Overall Tonality"]
        ]
        cols_conf = [
            "Article ID",
            "Medium",
            "Analysis By",
            "Analysis By Bureau",
            "Entity",
            "Category",
            "Sub Category1",
            "Overall Tonality",
            "Tonality",
            "Flag",
        ]
        render_excel_style_qc_section(conf_result_df, cols_conf, "conflicts_qc")
      else:
        st.success("✨ No tonality conflicts or missing values found!")

    # 6. Blank Tonality QC (Strictly Bureau is completely empty and Topic is present)
    with qc_tabs[5]:
      st.markdown("### ⚠️ Blank Tonality / Bureau QC")
      st.markdown(
          "<small style='color: #64748b;'>Showing strictly rows matching the rule: Bureau is strictly blank/empty and Topic is present.</small>",
          unsafe_allow_html=True,
      )
      
      blank_df = qc_df.copy()
      filtered_blank = []

      for idx, row in blank_df.iterrows():
        topic_val = row.get("Topic", "")
        bureau_val = row.get("Analysis By Bureau", "")

        has_topic = topic_val is not None and not pd.isna(topic_val) and str(topic_val).strip() != "" and str(topic_val).strip().lower() not in ["nan", "none", ""]
        
        # Strict blank check implementation as requested
        is_bureau_blank = (
            bureau_val is None 
            or pd.isna(bureau_val) 
            or str(bureau_val).strip() == ""
        )

        # Strict rule enforcement: Only keep rows where this rule is true
        if has_topic and is_bureau_blank:
          filtered_blank.append(row)

      if filtered_blank:
        blank_res_df = pd.DataFrame(filtered_blank)
        # Added the Bureau column ("Analysis By Bureau") along with all the other columns as requested
        cols_blank = [
            "Article ID",
            "Medium",
            "Analysis By",
            "Analysis By Bureau",
            "Overall Tonality",
            "Topic",
        ]
        render_excel_style_qc_section(blank_res_df, cols_blank, "blank_qc")
      else:
        st.success(
            "✨ No rows match the Blank Tonality / Bureau rule (Bureau strictly blank & Topic present)."
        )

  else:
    st.info(
        "ℹ️ Please upload an analysis spreadsheet above to activate the QC"
        " modules."
    )

else:
  st.markdown("<h1>🌍 Master Taxonomy Manager</h1>", unsafe_allow_html=True)
  st.markdown(
      "<p style='font-size: 16px; color: #475569;'>Seamlessly search, filter,"
      " analyze, and manage shared classification rules.</p>",
      unsafe_allow_html=True,
  )
  st.divider()

  col_btn1, col_btn2, _ = st.columns([1, 1, 4])
  with col_btn1:
    if st.button("➕ Add New Entry", use_container_width=True):
      add_taxonomy_modal(categories, subcategories, ALLOWED_TONALITY_OPTIONS)
  with col_btn2:
    if data:
      df_export = pd.DataFrame(data)
      cols_to_drop = [
          col for col in ["id", "created_at"] if col in df_export.columns
      ]
      df_export = df_export.drop(columns=cols_to_drop)
      csv_data = df_export.to_csv(index=False).encode("utf-8")
      st.download_button(
          label="📥 Export Master CSV",
          data=csv_data,
          file_name="Master_Taxonomy_Report.csv",
          mime="text/csv",
          use_container_width=True,
      )

  st.markdown("### 🔍 Search Database Filters")
  with st.container():
    s_col1, s_col2, s_col3 = st.columns(3)
    with s_col1:
      search_topic = st.selectbox("Topic Filter", options=["All"] + topics)
    with s_col2:
      search_category = st.selectbox(
          "Category Filter", options=["All"] + categories
      )
    with s_col3:
      search_subcategory = st.selectbox(
          "Subcategory Filter", options=["All"] + subcategories
      )

    f_col1, f_col2 = st.columns([2, 2])
    with f_col1:
      date_filter_option = st.selectbox(
          "📅 Filter by Creation/Update Date",
          options=[
              "All Time",
              "Today",
              "Two Weeks Old (Last 14 Days)",
              "One Month Old (Last 30 Days)",
              "Custom Range",
          ],
      )

    custom_date_range = None
    if date_filter_option == "Custom Range":
      with f_col2:
        custom_date_range = st.date_input(
            "Select Custom Date Range",
            value=(date.today() - timedelta(days=7), date.today()),
        )

  filtered_data = data
  if search_topic != "All":
    filtered_data = [
        item for item in filtered_data if item.get("topic") == search_topic
    ]
  if search_category != "All":
    filtered_data = [
        item for item in filtered_data if item.get("category") == search_category
    ]
  if search_subcategory != "All":
    filtered_data = [
        item
        for item in filtered_data
        if item.get("subcategory") == search_subcategory
    ]

  if date_filter_option != "All Time":
    now_utc = datetime.utcnow()
    filtered_by_date = []
    for item in filtered_data:
      created_at_str = item.get("created_at")
      if not created_at_str:
        continue
      try:
        created_dt = datetime.fromisoformat(
            created_at_str.replace("Z", "+00:00").split("+")[0]
        )
        item_date = created_dt.date()
        if date_filter_option == "Today":
          if item_date == now_utc.date():
            filtered_by_date.append(item)
        elif date_filter_option == "Two Weeks Old (Last 14 Days)":
          if now_utc - created_dt <= timedelta(days=14):
            filtered_by_date.append(item)
        elif date_filter_option == "One Month Old (Last 30 Days)":
          if now_utc - created_dt <= timedelta(days=30):
            filtered_by_date.append(item)
        elif date_filter_option == "Custom Range" and custom_date_range:
          if len(custom_date_range) == 2:
            start_d, end_d = custom_date_range
            if start_d <= item_date <= end_d:
              filtered_by_date.append(item)
      except Exception:
        pass
    filtered_data = filtered_by_date

  st.divider()

  res_col1, res_col2 = st.columns([4, 1])
  with res_col1:
    st.subheader("📋 Filtered Taxonomy Records")
  with res_col2:
    st.markdown(
        f"<div style='text-align: right; background-color: #e0f2fe;"
        f" color: #0369a1; padding: 6px 14px; border-radius: 20px; font-weight:"
        f" bold; font-size: 13px;'>📊 {len(filtered_data)} entries found</div>",
        unsafe_allow_html=True,
    )

  if not filtered_data:
    st.info("ℹ️ No matching taxonomy records found matching your criteria.")
  else:
    header_cols = st.columns([2, 2, 2, 2, 1.5, 1])
    with header_cols[0]:
      st.markdown("**Category**")
    with header_cols[1]:
      st.markdown("**Subcategory**")
    with header_cols[2]:
      st.markdown("**Topic**")
    with header_cols[3]:
      st.markdown("**Tonality Rule**")
    with header_cols[4]:
      st.markdown("**Submitted By**")
    with header_cols[5]:
      st.markdown("**Action**")

    st.divider()

    for idx, item in enumerate(filtered_data):
      row_cols = st.columns([2, 2, 2, 2, 1.5, 1])
      with row_cols[0]:
        st.write(item.get("category", ""))
      with row_cols[1]:
        st.write(
            item.get("subcategory") if item.get("subcategory") else "—"
        )
      with row_cols[2]:
        st.write(item.get("topic", ""))
      with row_cols[3]:
        st.write(item.get("tonality", ""))
      with row_cols[4]:
        st.write(item.get("submitted_by", "—"))
      with row_cols[5]:
        if st.button(
            "✏️ Edit",
            key=f"edit_btn_{item.get('id', idx)}",
            use_container_width=True,
        ):
          edit_taxonomy_modal(
              item, categories, subcategories, ALLOWED_TONALITY_OPTIONS
          )
      st.divider()
