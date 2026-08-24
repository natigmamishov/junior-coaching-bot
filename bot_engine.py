# ============================================================
# JUNIOR COACHING
# AI SALES & CONVERSATION ENGINE
# V10.1 - PLAYBOOK DRIVEN + APP.PY COMPATIBILITY
# ============================================================

import os
import re
import json
import uuid
import sqlite3
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
from dotenv import load_dotenv
from openai import OpenAI

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# 1. PATHS / CONFIG
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

FAQ_PATH = os.path.join(
    BASE_DIR,
    "Junior_Coaching_sesli_AI_FAQ.txt"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "junior_coaching.db"
)

load_dotenv(os.path.join(BASE_DIR, ".env"))

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)


# ============================================================
# 2. OPENAI CLIENT
# ============================================================

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY tapılmadı. "
        ".env və ya Streamlit Secrets daxilində əlavə edin."
    )

http_client = httpx.Client(
    verify=False,
    timeout=60.0
)

client = OpenAI(
    api_key=OPENAI_API_KEY,
    http_client=http_client
)


# ============================================================
# 3. BUSINESS FACTS
# ============================================================

BUSINESS_FACTS = {

    "program_name": "Junior Coaching",

    "age_min": 12,
    "age_max": 18,

    "near_age_rule": (
        "12 yaşa çox yaxın olan uşaqlar avtomatik rədd edilmir. "
        "Qısa görüntülü tanışlıqdan sonra mütəxəssis "
        "qrupa uyğunluğu qiymətləndirir."
    ),

    "format": (
        "Canlı qrup formatıdır. Praktik məşqlər, komanda işi, "
        "situasiyalar, layihələr və təqdimatlardan istifadə olunur."
    ),

    "frequency": "Ayda 3 bazar günü",

    "group_session_duration": "2 saat",

    "full_program_duration": "9 ay / 27 görüş",

    "language": (
        "Görüşlər əsasən Azərbaycan dilində keçirilir. "
        "Ehtiyac olduqda bəzi materiallar rus və ya ingilis "
        "dilində təqdim oluna bilər."
    ),

    "address": (
        "Süleyman Sani Axundov küçəsi, ADAS Plaza — "
        "ELİT T/M yaxınlığı"
    ),

    "individual_coaching_duration": "30–45 dəqiqə",

    "individual_coaching_price": 80,

    # Qrup modul qiymətləri hələ production source-of-truth deyil
    "foundation_price": None,
    "leadership_price": None,
    "pro_price": None,
    "impact_price": None,

    "child_intro_duration": "təxminən 5 dəqiqə",

    "parent_initial_call_duration": "5–7 dəqiqə",

    "therapy_boundary": (
        "Junior Coaching terapiya, psixoloji və ya "
        "psixiatrik müalicə xidməti deyil."
    )
}


# ============================================================
# 4. CONSTANTS
# ============================================================

SALES_STAGES = {
    "NEW",
    "DISCOVERY",
    "FIT_PRELIMINARY",
    "READY_TO_PROCEED",
    "CHILD_INTRO_PENDING",
    "CHILD_INTRO_BOOKED",
    "CHILD_INTRO_COMPLETED",
    "FIT_APPROVED",
    "FIT_NOT_APPROVED",
    "PAYMENT_PENDING",
    "REGISTERED",
    "HUMAN_HANDOFF"
}

CHILD_WILLINGNESS_VALUES = {
    "unknown",
    "willing",
    "hesitant",
    "unwilling"
}


# ============================================================
# 5. HELPERS
# ============================================================

def get_baku_time():
    return datetime.now(
        ZoneInfo("Asia/Baku")
    )


def now_string():
    return get_baku_time().strftime(
        "%Y-%m-%d %H:%M:%S"
    )


def safe_text(value):
    if value is None:
        return ""
    return str(value).strip()


def compact_spaces(text):
    return re.sub(
        r"\s+",
        " ",
        safe_text(text)
    ).strip()


def normalize_text(text):

    text = safe_text(text).lower()

    replacements = {
        "ə": "e",
        "ı": "i",
        "ö": "o",
        "ü": "u",
        "ş": "s",
        "ç": "c",
        "ğ": "g"
    }

    for a, b in replacements.items():
        text = text.replace(a, b)

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_phone(value):

    if not value:
        return None

    digits = re.sub(
        r"\D",
        "",
        str(value)
    )

    if len(digits) == 10 and digits.startswith("0"):
        return digits

    if len(digits) == 12 and digits.startswith("994"):
        return digits

    return None


def safe_int(value):

    try:

        if value is None:
            return None

        if isinstance(value, int):
            return value

        match = re.search(
            r"\b(\d{1,2})\b",
            str(value)
        )

        if not match:
            return None

        return int(match.group(1))

    except Exception:
        return None


def json_dumps(data):
    return json.dumps(
        data,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# 6. CHILD / LEAD STATE
# ============================================================

def create_empty_child():

    return {

        "child_id": str(uuid.uuid4()),

        "name": None,
        "age": None,

        "need": None,
        "need_tags": [],

        "context": None,
        "duration": None,
        "impact": None,
        "desired_outcome": None,

        "willingness": "unknown",

        "hypothesis": None,
        "hypothesis_confidence": None,

        "recommended_path": None,

        "discovery_question_count": 0,
        "discovery_complete": False
    }


def create_empty_lead(source="Unknown"):

    child = create_empty_child()

    return {

        "parent_name": None,
        "parent_title": None,
        "phone": None,

        "children": [child],
        "active_child_index": 0,

        # legacy compatibility
        "child_name": None,
        "child_age": None,
        "main_concern": None,

        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,

        # sales
        "sales_stage": "NEW",
        "ready_to_proceed": False,

        "child_intro_status": "NOT_STARTED",
        "fit_status": "UNKNOWN",
        "payment_status": "NOT_STARTED",

        "recommended_path": None,

        # booking
        "preferred_call_time": None,
        "agreed_followup_at": None,

        # ownership
        "handoff_status": "none",
        "owner": "AI",

        # conversation
        "primary_intent": None,
        "last_intents": [],
        "next_action": None,

        "_conversation_history": [],
        "_last_analysis": None,
        "_last_bot_response": None,
        "_last_user_message": None,
        "_last_question_topic": None,

        "_last_intent": None,
        "_last_primary_intent": None,
        "_last_confidence": None,
        "_last_faq_score": None,

        # old app
        "source": source,
        "status": "NEW"
    }


# ============================================================
# 7. CHILD HELPERS
# ============================================================

def ensure_children(lead):

    if "children" not in lead:
        lead["children"] = []

    if not lead["children"]:
        lead["children"].append(
            create_empty_child()
        )

    if lead.get("active_child_index") is None:
        lead["active_child_index"] = 0

    if lead["active_child_index"] >= len(
        lead["children"]
    ):
        lead["active_child_index"] = 0

    return lead["children"]


def get_active_child(lead):

    ensure_children(lead)

    index = lead.get(
        "active_child_index",
        0
    )

    return lead["children"][index]


def sync_legacy_fields(lead):

    child = get_active_child(lead)

    lead["child_name"] = child.get("name")
    lead["child_age"] = child.get("age")
    lead["main_concern"] = child.get("need")

    lead["concern_duration"] = child.get(
        "duration"
    )

    lead["concern_onset"] = child.get(
        "context"
    )

    lead["recommended_path"] = child.get(
        "recommended_path"
    )

    return lead


def find_child_by_name(
    lead,
    name
):

    if not name:
        return None

    target = normalize_text(name)

    for i, child in enumerate(
        lead.get("children", [])
    ):

        child_name = normalize_text(
            child.get("name")
        )

        if (
            child_name
            and child_name == target
        ):
            return i

    return None


def create_new_child(lead):

    child = create_empty_child()

    lead["children"].append(
        child
    )

    index = len(
        lead["children"]
    ) - 1

    lead["active_child_index"] = index

    sync_legacy_fields(lead)

    return index


# ============================================================
# 8. FAQ / KNOWLEDGE BASE
# ============================================================

FAQ_ITEMS = []
FAQ_VECTORIZER = None
FAQ_MATRIX = None


def parse_faq_file():

    if not os.path.exists(
        FAQ_PATH
    ):
        return []

    with open(
        FAQ_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        text = f.read()

    pattern = re.compile(
        r"""
        (?:^|\n)
        \s*\d+\.\s*
        Sual:\s*
        (.*?)
        \n
        \s*
        (?:Agent|Selnaz|Cavab):
        \s*
        (.*?)
        (?=
            \n\s*\d+\.\s*Sual:
            |
            \Z
        )
        """,
        re.S | re.X | re.I
    )

    items = []

    for match in pattern.finditer(
        text
    ):

        question = compact_spaces(
            match.group(1)
        )

        answer = compact_spaces(
            match.group(2)
        )

        if question and answer:

            items.append({
                "question": question,
                "answer": answer
            })

    return items


def build_faq_index():

    global FAQ_ITEMS
    global FAQ_VECTORIZER
    global FAQ_MATRIX

    FAQ_ITEMS = parse_faq_file()

    if not FAQ_ITEMS:

        FAQ_VECTORIZER = None
        FAQ_MATRIX = None

        return

    corpus = [
        item["question"]
        for item in FAQ_ITEMS
    ]

    FAQ_VECTORIZER = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=50000
    )

    FAQ_MATRIX = FAQ_VECTORIZER.fit_transform(
        corpus
    )


def retrieve_similar(
    query,
    top_k=5
):

    if (
        FAQ_VECTORIZER is None
        or FAQ_MATRIX is None
        or not FAQ_ITEMS
    ):
        return []

    q_vec = FAQ_VECTORIZER.transform(
        [query]
    )

    scores = cosine_similarity(
        q_vec,
        FAQ_MATRIX
    )[0]

    ranked_indices = scores.argsort()[::-1]

    results = []

    for idx in ranked_indices[:top_k]:

        results.append({

            "question": FAQ_ITEMS[idx][
                "question"
            ],

            "answer": FAQ_ITEMS[idx][
                "answer"
            ],

            "score": float(
                scores[idx]
            )
        })

    return results


def get_best_faq_hit(
    query,
    min_score=0.25
):

    hits = retrieve_similar(
        query,
        top_k=1
    )

    if not hits:
        return None

    if hits[0]["score"] < min_score:
        return None

    return hits[0]


build_faq_index()


# ============================================================
# 9. DATABASE
# ============================================================

def get_connection():

    conn = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    conn.row_factory = sqlite3.Row

    return conn


def add_column_if_missing(
    conn,
    table_name,
    column_name,
    column_definition
):

    rows = conn.execute(
        f"PRAGMA table_info({table_name})"
    ).fetchall()

    columns = {
        row[1]
        for row in rows
    }

    if column_name not in columns:

        conn.execute(
            f"""
            ALTER TABLE {table_name}
            ADD COLUMN {column_name}
            {column_definition}
            """
        )


def init_db():

    conn = get_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS leads (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            parent_name TEXT,
            parent_title TEXT,

            child_name TEXT,
            child_age INTEGER,

            main_concern TEXT,

            phone TEXT,
            preferred_call_time TEXT,

            source TEXT,
            status TEXT,

            created_at TEXT,
            updated_at TEXT,

            needs_concern_followup INTEGER DEFAULT 0,
            concern_duration TEXT,
            concern_onset TEXT
        )
        """
    )

    new_lead_columns = {

        "children_json":
            "TEXT",

        "sales_stage":
            "TEXT",

        "ready_to_proceed":
            "INTEGER DEFAULT 0",

        "child_intro_status":
            "TEXT",

        "fit_status":
            "TEXT",

        "payment_status":
            "TEXT",

        "recommended_path":
            "TEXT",

        "handoff_status":
            "TEXT",

        "owner":
            "TEXT",

        "agreed_followup_at":
            "TEXT",

        "next_action":
            "TEXT",

        "conversation_summary":
            "TEXT"
    }

    for column, definition in new_lead_columns.items():

        add_column_if_missing(
            conn,
            "leads",
            column,
            definition
        )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_logs (

            id INTEGER PRIMARY KEY AUTOINCREMENT,

            session_id TEXT,

            user_message TEXT,
            bot_response TEXT,

            intent TEXT,
            confidence REAL,

            faq_score REAL,
            current_field TEXT,

            parent_name TEXT,
            parent_title TEXT,

            child_name TEXT,
            child_age INTEGER,
            main_concern TEXT,

            phone TEXT,
            preferred_call_time TEXT,

            status TEXT,
            source TEXT,

            state_json TEXT,
            analysis_json TEXT,

            created_at TEXT
        )
        """
    )

    add_column_if_missing(
        conn,
        "conversation_logs",
        "state_json",
        "TEXT"
    )

    add_column_if_missing(
        conn,
        "conversation_logs",
        "analysis_json",
        "TEXT"
    )

    conn.commit()
    conn.close()


init_db()


# ============================================================
# 10. DATABASE CRUD
# ============================================================

def find_lead_by_phone(
    phone
):

    phone = normalize_phone(
        phone
    )

    if not phone:
        return None

    conn = get_connection()

    row = conn.execute(
        """
        SELECT *
        FROM leads
        WHERE phone = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (phone,)
    ).fetchone()

    conn.close()

    return (
        dict(row)
        if row
        else None
    )


def build_conversation_summary(
    lead
):

    parts = []

    if lead.get("parent_name"):

        parts.append(
            f"Valideyn: {lead['parent_name']}"
        )

    if lead.get("phone"):

        parts.append(
            f"Telefon: {lead['phone']}"
        )

    for i, child in enumerate(
        lead.get(
            "children",
            []
        ),
        start=1
    ):

        child_parts = []

        if child.get("name"):

            child_parts.append(
                f"ad={child['name']}"
            )

        if child.get("age") is not None:

            child_parts.append(
                f"yaş={child['age']}"
            )

        if child.get("need"):

            child_parts.append(
                f"ehtiyac={child['need']}"
            )

        if child.get(
            "desired_outcome"
        ):

            child_parts.append(
                "istənilən nəticə="
                +
                child["desired_outcome"]
            )

        if child_parts:

            parts.append(
                f"Uşaq {i}: "
                +
                ", ".join(
                    child_parts
                )
            )

    if lead.get(
        "recommended_path"
    ):

        parts.append(
            "İlkin tövsiyə: "
            +
            str(
                lead[
                    "recommended_path"
                ]
            )
        )

    if lead.get("next_action"):

        parts.append(
            "Növbəti addım: "
            +
            str(
                lead["next_action"]
            )
        )

    return " | ".join(
        parts
    )


def save_lead_to_db(
    lead
):

    sync_legacy_fields(
        lead
    )

    conn = get_connection()

    created_at = now_string()

    cursor = conn.execute(
        """
        INSERT INTO leads (

            parent_name,
            parent_title,

            child_name,
            child_age,

            main_concern,

            phone,
            preferred_call_time,

            source,
            status,

            created_at,
            updated_at,

            needs_concern_followup,
            concern_duration,
            concern_onset,

            children_json,
            sales_stage,
            ready_to_proceed,
            child_intro_status,
            fit_status,
            payment_status,
            recommended_path,
            handoff_status,
            owner,
            agreed_followup_at,
            next_action,
            conversation_summary
        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,

        (
            lead.get("parent_name"),
            lead.get("parent_title"),

            lead.get("child_name"),
            lead.get("child_age"),

            lead.get("main_concern"),

            lead.get("phone"),
            lead.get(
                "preferred_call_time"
            ),

            lead.get("source"),
            lead.get("status"),

            created_at,
            created_at,

            int(
                bool(
                    lead.get(
                        "needs_concern_followup"
                    )
                )
            ),

            lead.get(
                "concern_duration"
            ),

            lead.get(
                "concern_onset"
            ),

            json_dumps(
                lead.get(
                    "children",
                    []
                )
            ),

            lead.get(
                "sales_stage"
            ),

            int(
                bool(
                    lead.get(
                        "ready_to_proceed"
                    )
                )
            ),

            lead.get(
                "child_intro_status"
            ),

            lead.get(
                "fit_status"
            ),

            lead.get(
                "payment_status"
            ),

            lead.get(
                "recommended_path"
            ),

            lead.get(
                "handoff_status"
            ),

            lead.get("owner"),

            lead.get(
                "agreed_followup_at"
            ),

            lead.get(
                "next_action"
            ),

            build_conversation_summary(
                lead
            )
        )
    )

    lead_id = cursor.lastrowid

    conn.commit()
    conn.close()

    return lead_id


# ============================================================
# 11. STATE FOR LLM
# ============================================================

def sanitize_state_for_llm(
    lead
):

    keys = [

        "parent_name",
        "parent_title",
        "phone",

        "children",
        "active_child_index",

        "sales_stage",
        "ready_to_proceed",

        "child_intro_status",
        "fit_status",
        "payment_status",

        "recommended_path",

        "preferred_call_time",
        "agreed_followup_at",

        "handoff_status",
        "owner",

        "primary_intent",
        "next_action"
    ]

    result = {}

    for key in keys:

        result[key] = deepcopy(
            lead.get(key)
        )

    return result


# ============================================================
# 12. CONVERSATION HISTORY
# ============================================================

def add_history(
    lead,
    role,
    content
):

    if "_conversation_history" not in lead:

        lead[
            "_conversation_history"
        ] = []

    lead[
        "_conversation_history"
    ].append({

        "role": role,
        "content": content
    })

    lead[
        "_conversation_history"
    ] = (
        lead[
            "_conversation_history"
        ][-16:]
    )


def recent_history_text(
    lead,
    limit=10
):

    history = lead.get(
        "_conversation_history",
        []
    )[-limit:]

    lines = []

    for item in history:

        role = item.get("role")

        label = (
            "Valideyn"
            if role == "user"
            else "Leyla"
        )

        lines.append(
            f"{label}: "
            f"{item.get('content', '')}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# 13. LLM JSON HELPER
# ============================================================

def call_json_llm(
    system_prompt,
    user_prompt,
    temperature=0
):

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            temperature=temperature,

            response_format={
                "type": "json_object"
            },

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ]
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        return json.loads(
            content
        )

    except Exception as exc:

        print(
            "LLM JSON ERROR:",
            repr(exc)
        )

        return None


# ============================================================
# 14. ANALYZER
# ============================================================

ANALYSIS_SYSTEM_PROMPT = """
Sən Junior Coaching üçün conversation-understanding modulusan.

Sən istifadəçiyə cavab yazmırsan.
Sən mesajı strukturlaşdırırsan.

Prioritet:
1. Safety / risk
2. Primary intent
3. Explicit corrections
4. Explicit facts
5. Current state
6. Secondary intents
7. Need reasoning
8. Next-step signal

Qaydalar:

- Bir mesajda bir neçə məlumat varsa hamısını çıxar.
- Bir mesajda bir neçə sual varsa hamısını çıxar.
- Adı təxmin etmə.
- "anasıyam", "mənə", "bu nömrə ilə", "maraqlanıram"
  kimi ifadələri ad kimi götürmə.
- "Mən Aygünəm" -> parent name Aygün.
- "Adım İsmayıldır" -> parent name İsmayıl.
- "İsmayıl bəy" -> parent name İsmayıl, title bəy.
- Uşağın adını valideyn adı ilə qarışdırma.
- "Tunar yox, Turandır" correction.
- "16 yox, 15 yaşı var" correction.
- "kiçik oğlum", "digər qızım" explicit yeni uşaq ola bilər.
- Amma heç bir explicit ikinci uşaq siqnalı yoxdursa yeni child yaratma.
- Davranış təsvirindən need çıxara bilərsən.
- Diaqnoz qoyma.
- "özgüvənsizlik", "məqsəd və gələcək", "məsuliyyətsizdir"
  kimi ifadələr need ola bilər.
- "başlamaq istəyirik", "qeydiyyata keçək",
  "bizə uyğundur", "davam edək", "razıyıq"
  ready-to-proceed siqnalıdır.
- "cümə günü cavab verərəm", "sabah yoldaşımla danışacağam"
  follow-up commitment ola bilər.
- "harada", "qiymət", "hansı gün", "neçə dəqiqə"
  ayrıca question topic-dir.
- State recall:
  "adımı necə qeyd etmisiniz?"
  "uşağın yaşı neçə idi?"
  "neçə övlad demişdim?"
- Primary intent extraction faktından daha vacibdir.
- Yeni məlumat əvvəlki hypothesis-i təkzib edirsə
  contradicts_previous_hypothesis=true et.
- Need artıq aydındırsa clarification_needed=false.
- Minimum question -> maximum insight.

JSON-dan başqa heç nə yazma.
"""


def analyze_user_message(
    user_text,
    lead
):

    faq_hits = retrieve_similar(
        user_text,
        top_k=6
    )

    faq_context = []

    for hit in faq_hits:

        faq_context.append({

            "question":
                hit["question"],

            "answer":
                hit["answer"],

            "score":
                round(
                    hit["score"],
                    3
                )
        })

    state = sanitize_state_for_llm(
        lead
    )

    current_date = get_baku_time().strftime(
        "%Y-%m-%d"
    )

    user_prompt = f"""
CURRENT DATE:
{current_date}

CURRENT STRUCTURED STATE:
{json_dumps(state)}

RECENT CONVERSATION:
{recent_history_text(lead)}

CURRENT USER MESSAGE:
{user_text}

TOP KNOWLEDGE BASE CANDIDATES:
{json_dumps(faq_context)}

Bu JSON-u qaytar:

{{
  "primary_intent": "...",

  "all_intents": [],

  "confidence": 0.0,

  "is_question": false,

  "questions": [
    {{
      "topic": "...",
      "question": "..."
    }}
  ],

  "parent": {{
    "name": null,
    "title": null,
    "phone": null
  }},

  "children": [
    {{
      "target": "active|new|unknown",
      "name": null,
      "age": null,
      "need": null,
      "need_tags": [],
      "context": null,
      "duration": null,
      "impact": null,
      "desired_outcome": null,
      "willingness": null,
      "explicit_new_child": false
    }}
  ],

  "corrections": [
    {{
      "field": "parent_name|child_name|child_age|need|phone|other",
      "old_value": null,
      "new_value": null,
      "child_reference": null
    }}
  ],

  "need_analysis": {{
    "need_is_clear": false,
    "need_summary": null,
    "need_tags": [],
    "hypothesis": null,
    "hypothesis_confidence": 0.0,
    "contradicts_previous_hypothesis": false,
    "clarification_needed": false,
    "best_clarification_question": null,
    "recommendation_signal": "personal_social|future_direction|mixed|unknown"
  }},

  "ready_to_proceed": false,

  "child_intro_requested": false,

  "human_requested": false,

  "clinical_or_safety_risk": false,

  "complaint": false,

  "partnership": false,

  "special_payment_request": false,

  "agreed_followup_at": null,

  "preferred_contact_time": null,

  "state_recall": {{
    "requested": false,
    "fields": []
  }},

  "conversation_act":
    "greeting|question|answer|correction|objection|commitment|information|other"
}}

primary_intent nümunələri:

greeting
program_info
eligibility
price
location
schedule
duration
language
child_resistance
consultation
provide_information
state_recall
ready_to_proceed
followup_commitment
human_request
complaint
clinical_risk
partnership
special_payment
registration
other

Məsələn istifadəçi deyirsə:

"Oğlum evdə rahat danışır,
amma məktəbdə müəllim sual verəndə
bildiyi halda əl qaldırmır.
Bilmirəm özgüvənsizlikdir, yoxsa xarakteridir."

Primary intent:
consultation

Need:
məktəb mühitində özünüifadə / iştirak çətinliyi

Hypothesis:
bu vəziyyət sosial mühitdə özünüifadə və ya
situativ özgüvənlə əlaqəli ola bilər,
amma bunun ümumi xarakter xüsusiyyəti olduğunu
və ya problem olduğunu qəti demək olmaz.

Diaqnoz qoyma.
"""

    result = call_json_llm(

        ANALYSIS_SYSTEM_PROMPT,

        user_prompt,

        temperature=0
    )

    if not result:

        return fallback_analysis(
            user_text
        )

    return result


# ============================================================
# 15. FALLBACK ANALYZER
# ============================================================

def fallback_analysis(
    user_text
):

    normalized = normalize_text(
        user_text
    )

    intent = (
        "provide_information"
    )

    if normalized in {
        "salam",
        "slm",
        "salamlar"
    }:

        intent = "greeting"

    elif any(
        x in normalized
        for x in [
            "qiymet",
            "ne qeder"
        ]
    ):

        intent = "price"

    elif any(
        x in normalized
        for x in [
            "harada",
            "unvan",
            "adres",
            "mekan"
        ]
    ):

        intent = "location"

    return {

        "primary_intent": intent,

        "all_intents": [
            intent
        ],

        "confidence": 0.4,

        "is_question":
            "?" in user_text,

        "questions": [],

        "parent": {
            "name": None,
            "title": None,
            "phone": None
        },

        "children": [],

        "corrections": [],

        "need_analysis": {

            "need_is_clear":
                False,

            "need_summary":
                None,

            "need_tags":
                [],

            "hypothesis":
                None,

            "hypothesis_confidence":
                0.0,

            "contradicts_previous_hypothesis":
                False,

            "clarification_needed":
                False,

            "best_clarification_question":
                None,

            "recommendation_signal":
                "unknown"
        },

        "ready_to_proceed":
            False,

        "child_intro_requested":
            False,

        "human_requested":
            False,

        "clinical_or_safety_risk":
            False,

        "complaint":
            False,

        "partnership":
            False,

        "special_payment_request":
            False,

        "agreed_followup_at":
            None,

        "preferred_contact_time":
            None,

        "state_recall": {
            "requested": False,
            "fields": []
        },

        "conversation_act":
            "other"
    }


# ============================================================
# 16. CORRECTIONS
# ============================================================

def apply_corrections(
    lead,
    analysis
):

    corrections = (
        analysis.get(
            "corrections",
            []
        )
        or []
    )

    for correction in corrections:

        field = correction.get(
            "field"
        )

        new_value = correction.get(
            "new_value"
        )

        child_reference = correction.get(
            "child_reference"
        )

        if field == "parent_name":

            if new_value:

                lead[
                    "parent_name"
                ] = compact_spaces(
                    new_value
                )

        elif field == "phone":

            phone = normalize_phone(
                new_value
            )

            if phone:
                lead["phone"] = phone

        elif field in {
            "child_name",
            "child_age",
            "need"
        }:

            target_index = None

            if child_reference:

                target_index = (
                    find_child_by_name(
                        lead,
                        child_reference
                    )
                )

            if target_index is None:

                target_index = (
                    lead.get(
                        "active_child_index",
                        0
                    )
                )

            ensure_children(
                lead
            )

            child = lead[
                "children"
            ][target_index]

            if field == "child_name":

                if new_value:

                    child[
                        "name"
                    ] = compact_spaces(
                        new_value
                    )

            elif field == "child_age":

                age = safe_int(
                    new_value
                )

                if age is not None:

                    child[
                        "age"
                    ] = age

            elif field == "need":

                if new_value:

                    child[
                        "need"
                    ] = compact_spaces(
                        new_value
                    )

                    child[
                        "hypothesis"
                    ] = None

                    child[
                        "hypothesis_confidence"
                    ] = None

                    child[
                        "recommended_path"
                    ] = None

    sync_legacy_fields(
        lead
    )


# ============================================================
# 17. APPLY FACTS
# ============================================================

def apply_parent_facts(
    lead,
    analysis
):

    parent = (
        analysis.get(
            "parent",
            {}
        )
        or {}
    )

    parent_name = parent.get(
        "name"
    )

    parent_title = parent.get(
        "title"
    )

    phone = normalize_phone(
        parent.get(
            "phone"
        )
    )

    if parent_name:

        bad_names = {
            "ana",
            "anasi",
            "anasiyam",
            "mene",
            "men",
            "bu nomre",
            "nomre",
            "maraqlaniram"
        }

        normalized = normalize_text(
            parent_name
        )

        if normalized not in bad_names:

            lead[
                "parent_name"
            ] = compact_spaces(
                parent_name
            )

    if parent_title:

        norm_title = normalize_text(
            parent_title
        )

        if norm_title == "bey":

            lead[
                "parent_title"
            ] = "bəy"

        elif norm_title == "xanim":

            lead[
                "parent_title"
            ] = "xanım"

    if phone:

        lead["phone"] = phone


def apply_children_facts(
    lead,
    analysis
):

    extracted_children = (
        analysis.get(
            "children",
            []
        )
        or []
    )

    if not extracted_children:
        return

    ensure_children(
        lead
    )

    for item in extracted_children:

        explicit_new = bool(
            item.get(
                "explicit_new_child"
            )
        )

        target = item.get(
            "target",
            "active"
        )

        incoming_name = item.get(
            "name"
        )

        child_index = None

        if incoming_name:

            child_index = (
                find_child_by_name(
                    lead,
                    incoming_name
                )
            )

        if (
            child_index is None
            and
            (
                explicit_new
                or target == "new"
            )
        ):

            child_index = create_new_child(
                lead
            )

        if child_index is None:

            child_index = lead.get(
                "active_child_index",
                0
            )

        child = lead[
            "children"
        ][child_index]

        if incoming_name:

            child[
                "name"
            ] = compact_spaces(
                incoming_name
            )

        age = safe_int(
            item.get(
                "age"
            )
        )

        if age is not None:

            child[
                "age"
            ] = age

        need = item.get(
            "need"
        )

        if need:

            child[
                "need"
            ] = compact_spaces(
                need
            )

        tags = (
            item.get(
                "need_tags",
                []
            )
            or []
        )

        if tags:

            existing = set(
                child.get(
                    "need_tags",
                    []
                )
            )

            existing.update(
                tags
            )

            child[
                "need_tags"
            ] = list(
                existing
            )

        for field in [
            "context",
            "duration",
            "impact",
            "desired_outcome"
        ]:

            value = item.get(
                field
            )

            if value:

                child[
                    field
                ] = compact_spaces(
                    value
                )

        willingness = item.get(
            "willingness"
        )

        if (
            willingness
            in CHILD_WILLINGNESS_VALUES
        ):

            child[
                "willingness"
            ] = willingness

        lead[
            "active_child_index"
        ] = child_index

    sync_legacy_fields(
        lead
    )


def apply_need_analysis(
    lead,
    analysis
):

    need_analysis = (
        analysis.get(
            "need_analysis",
            {}
        )
        or {}
    )

    child = get_active_child(
        lead
    )

    summary = need_analysis.get(
        "need_summary"
    )

    if (
        summary
        and
        not child.get("need")
    ):

        child[
            "need"
        ] = compact_spaces(
            summary
        )

    tags = (
        need_analysis.get(
            "need_tags",
            []
        )
        or []
    )

    if tags:

        existing = set(
            child.get(
                "need_tags",
                []
            )
        )

        existing.update(
            tags
        )

        child[
            "need_tags"
        ] = list(
            existing
        )

    contradiction = bool(
        need_analysis.get(
            "contradicts_previous_hypothesis"
        )
    )

    if contradiction:

        child[
            "hypothesis"
        ] = None

        child[
            "hypothesis_confidence"
        ] = None

        child[
            "recommended_path"
        ] = None

    hypothesis = (
        need_analysis.get(
            "hypothesis"
        )
    )

    if hypothesis:

        child[
            "hypothesis"
        ] = compact_spaces(
            hypothesis
        )

        child[
            "hypothesis_confidence"
        ] = (
            need_analysis.get(
                "hypothesis_confidence"
            )
        )

    if need_analysis.get(
        "need_is_clear"
    ):

        child[
            "discovery_complete"
        ] = True

    sync_legacy_fields(
        lead
    )


def apply_other_facts(
    lead,
    analysis
):

    preferred_time = (
        analysis.get(
            "preferred_contact_time"
        )
    )

    if preferred_time:

        lead[
            "preferred_call_time"
        ] = compact_spaces(
            preferred_time
        )

    agreed_followup = (
        analysis.get(
            "agreed_followup_at"
        )
    )

    if agreed_followup:

        lead[
            "agreed_followup_at"
        ] = compact_spaces(
            agreed_followup
        )

    if analysis.get(
        "ready_to_proceed"
    ):

        lead[
            "ready_to_proceed"
        ] = True

        lead[
            "sales_stage"
        ] = "READY_TO_PROCEED"

    lead[
        "primary_intent"
    ] = analysis.get(
        "primary_intent"
    )

    lead[
        "_last_intent"
    ] = analysis.get(
        "primary_intent"
    )

    lead[
        "_last_primary_intent"
    ] = analysis.get(
        "primary_intent"
    )

    lead[
        "_last_confidence"
    ] = analysis.get(
        "confidence"
    )

    lead[
        "last_intents"
    ] = (
        analysis.get(
            "all_intents",
            []
        )
        or []
    )


def apply_analysis_to_state(
    lead,
    analysis
):

    apply_corrections(
        lead,
        analysis
    )

    apply_parent_facts(
        lead,
        analysis
    )

    apply_children_facts(
        lead,
        analysis
    )

    apply_need_analysis(
        lead,
        analysis
    )

    apply_other_facts(
        lead,
        analysis
    )

    sync_legacy_fields(
        lead
    )


# ============================================================
# 18. BUSINESS RULE ENGINE
# ============================================================

def evaluate_age_rule(
    child
):

    age = child.get(
        "age"
    )

    if age is None:

        return {
            "status": "UNKNOWN",
            "path": None
        }

    if 12 <= age <= 18:

        return {
            "status": "FIT_RANGE",
            "path": "GROUP"
        }

    if age == 11:

        return {
            "status": "SPECIALIST_REVIEW",
            "path": "SPECIALIST_REVIEW"
        }

    if age >= 19:

        return {
            "status": "INDIVIDUAL",
            "path": "INDIVIDUAL"
        }

    return {
        "status": "NOT_FIT_GROUP",
        "path": "SPECIALIST_REVIEW"
    }


def calculate_recommendation(
    lead,
    analysis=None
):

    child = get_active_child(
        lead
    )

    age_rule = evaluate_age_rule(
        child
    )

    if (
        age_rule[
            "status"
        ]
        == "INDIVIDUAL"
    ):

        child[
            "recommended_path"
        ] = "INDIVIDUAL"

        lead[
            "recommended_path"
        ] = "INDIVIDUAL"

        return "INDIVIDUAL"

    if age_rule[
        "status"
    ] in {
        "SPECIALIST_REVIEW",
        "NOT_FIT_GROUP"
    }:

        child[
            "recommended_path"
        ] = "SPECIALIST_REVIEW"

        lead[
            "recommended_path"
        ] = "SPECIALIST_REVIEW"

        return "SPECIALIST_REVIEW"

    signal = "unknown"

    if analysis:

        need_analysis = (
            analysis.get(
                "need_analysis",
                {}
            )
            or {}
        )

        signal = (
            need_analysis.get(
                "recommendation_signal",
                "unknown"
            )
        )

    if signal == "personal_social":

        recommendation = (
            "5_MONTH"
        )

    elif signal == "future_direction":

        recommendation = (
            "9_MONTH"
        )

    elif signal == "mixed":

        recommendation = None

    else:

        recommendation = child.get(
            "recommended_path"
        )

    child[
        "recommended_path"
    ] = recommendation

    lead[
        "recommended_path"
    ] = recommendation

    return recommendation


# ============================================================
# 19. HUMAN OWNERSHIP
# ============================================================

def human_owns_lead(
    lead
):

    return (
        lead.get("owner")
        == "HUMAN"

        or

        lead.get(
            "handoff_status"
        )
        == "assigned"
    )


def mark_handoff(
    lead,
    requested=True
):

    if requested:

        lead[
            "handoff_status"
        ] = "requested"

    lead[
        "sales_stage"
    ] = "HUMAN_HANDOFF"

    lead[
        "next_action"
    ] = "HUMAN_CONTACT"

    lead[
        "status"
    ] = "ESCALATED"


# ============================================================
# 20. STATE RECALL
# ============================================================

def answer_state_recall(
    lead,
    fields=None
):

    fields = fields or []

    child = get_active_child(
        lead
    )

    responses = []

    fields_norm = {
        normalize_text(x)
        for x in fields
    }

    if not fields_norm:

        if lead.get(
            "parent_name"
        ):

            responses.append(
                f"Adınızı {lead['parent_name']} kimi qeyd etmişəm."
            )

        if child.get(
            "name"
        ):

            responses.append(
                f"Övladınızın adı {child['name']}-dır."
            )

        if child.get(
            "age"
        ) is not None:

            responses.append(
                f"Yaşı {child['age']} olaraq qeyd olunub."
            )

        if child.get(
            "need"
        ):

            responses.append(
                f"Əsas ehtiyac kimi “{child['need']}” qeyd olunub."
            )

        if responses:

            return " ".join(
                responses
            )

        return (
            "Hazırda bu məlumatlar tam qeyd olunmayıb."
        )

    if any(
        "parent" in x
        or x == "ad"
        for x in fields_norm
    ):

        if lead.get(
            "parent_name"
        ):

            responses.append(
                f"Adınızı {lead['parent_name']} kimi qeyd etmişəm."
            )

        else:

            responses.append(
                "Adınız hələ qeyd olunmayıb."
            )

    if any(
        "child_name" in x
        or "usaq adi" in x
        or "ovladin adi" in x
        for x in fields_norm
    ):

        if child.get(
            "name"
        ):

            responses.append(
                f"Övladınızın adı {child['name']}-dır."
            )

        else:

            responses.append(
                "Övladınızın adı hələ qeyd olunmayıb."
            )

    if any(
        "age" in x
        or "yas" in x
        for x in fields_norm
    ):

        if child.get(
            "age"
        ) is not None:

            responses.append(
                f"Yaşını {child['age']} olaraq qeyd etmişəm."
            )

        else:

            responses.append(
                "Yaşı hələ qeyd olunmayıb."
            )

    if any(
        "need" in x
        or "concern" in x
        or "ehtiyac" in x
        or "narahat" in x
        for x in fields_norm
    ):

        if child.get(
            "need"
        ):

            responses.append(
                f"Əsas ehtiyac kimi “{child['need']}” qeyd olunub."
            )

        else:

            responses.append(
                "Əsas ehtiyac hələ qeyd olunmayıb."
            )

    if any(
        "child_count" in x
        or "usaq sayi" in x
        or "ovlad sayi" in x
        for x in fields_norm
    ):

        count = len([
            c
            for c in lead.get(
                "children",
                []
            )
            if (
                c.get("name")
                or c.get("age") is not None
                or c.get("need")
            )
        ])

        if count:

            responses.append(
                f"Hazırda {count} övlad üzrə məlumat qeyd olunub."
            )

        else:

            responses.append(
                "Övlad sayı barədə dəqiq məlumat hələ yoxdur."
            )

    return " ".join(
        responses
    )


# ============================================================
# 21. BUSINESS FACT ANSWERS
# ============================================================

def answer_location():

    return (
        "Görüşlər "
        + BUSINESS_FACTS[
            "address"
        ]
        + " ünvanında keçirilir."
    )


def answer_schedule():

    return (
        "Qrup görüşləri "
        + BUSINESS_FACTS[
            "frequency"
        ]
        + " keçirilir. Dəqiq tarix və saatlar "
          "əvvəlcədən valideynlərlə paylaşılır."
    )


def answer_duration():

    return (
        "Qrup görüşü "
        + BUSINESS_FACTS[
            "group_session_duration"
        ]
        + " davam edir. Valideynlə ilkin tanışlıq zəngi "
        "isə adətən "
        + BUSINESS_FACTS[
            "parent_initial_call_duration"
        ]
        + " olur."
    )


def answer_language():

    return BUSINESS_FACTS[
        "language"
    ]


def answer_price(
    lead
):

    child = get_active_child(
        lead
    )

    age = child.get(
        "age"
    )

    if (
        age is not None
        and age >= 19
    ):

        return (
            "19 yaş və yuxarı üçün fərdi coaching "
            f"{BUSINESS_FACTS['individual_coaching_duration']} "
            f"davam edir və bir görüş "
            f"{BUSINESS_FACTS['individual_coaching_price']} AZN-dir."
        )

    return (
        "Qrup proqramında ödəniş modul üzrə edilir. "
        "Modulların dəqiq məbləğləri təsdiqlənmiş source-da "
        "olmadığı üçün sizə rəqəm uydurmaq istəmirəm."
    )


def answer_eligibility(
    lead
):

    child = get_active_child(
        lead
    )

    age = child.get(
        "age"
    )

    if age is None:

        return (
            "Junior Coaching-in əsas qrupu 12–18 yaş üçündür."
        )

    if 12 <= age <= 18:

        return (
            f"{age} yaş Junior Coaching-in "
            "12–18 yaş qrupuna uyğundur."
        )

    if age == 11:

        return (
            "12 yaşa yaxın olduğu üçün avtomatik "
            "uyğun deyil demirik. "
            "Qısa görüntülü tanışlıqdan sonra "
            "mütəxəssis qrupa uyğunluğu qiymətləndirə bilər."
        )

    if age >= 19:

        return (
            "Junior Coaching-in qrup formatı "
            "12–18 yaş üçündür. "
            "19 yaş və yuxarı üçün fərdi coaching "
            "daha uyğun seçimdir."
        )

    return (
        "Əsas Junior Coaching qrupu 12–18 yaş üçündür. "
        "Bu yaş üçün uyğunluğu mütəxəssis ayrıca "
        "qiymətləndirməlidir."
    )


# ============================================================
# 22. READY TO PROCEED
# ============================================================

def handle_ready_to_proceed(
    lead
):

    lead[
        "ready_to_proceed"
    ] = True

    lead[
        "sales_stage"
    ] = "CHILD_INTRO_PENDING"

    lead[
        "child_intro_status"
    ] = "PENDING"

    lead[
        "next_action"
    ] = "BOOK_CHILD_INTRO"

    if lead.get(
        "preferred_call_time"
    ):

        lead[
            "child_intro_status"
        ] = "BOOKED"

        lead[
            "sales_stage"
        ] = "CHILD_INTRO_BOOKED"

        lead[
            "next_action"
        ] = "CHILD_INTRO"

        lead[
            "status"
        ] = "CALL_REQUESTED"

        return (
            "Əla. Növbəti addım övladınızla "
            "qısa görüntülü tanışlıqdır. "
            f"{lead['preferred_call_time']} üçün "
            "müraciətinizi qeyd etdim ✅"
        )

    return (
        "Əla. Növbəti addım övladınızla "
        "qısa görüntülü tanışlıqdır. "
        "Sizə uyğun gün və saat aralığını yaza bilərsiniz?"
    )


# ============================================================
# 23. RESPONSE GENERATOR
# ============================================================

RESPONSE_SYSTEM_PROMPT = """
Sən Junior Coaching üzrə virtual bələdçi və
AI Sales Assistant Leylasan.

FAQ bot kimi yox, trusted advisor kimi davran.

Prioritet:
1. Safety / Business Rules
2. Primary intent
3. Current structured state
4. Conversation history
5. Knowledge base
6. Contextual reasoning
7. Next step

Qaydalar:

- Əvvəl istifadəçinin əsas sualını cavablandır.
- State update-i lazımsız şəkildə istifadəçiyə demə.
- Bir mesajda bir neçə sual varsa hamısını nəzərə al.
- Knowledge base-də olmayan biznes faktını uydurma.
- Qiymət, endirim, tarix, kampaniya uydurma.
- Correction varsa yeni məlumatı əsas götür.
- Yeni məlumat əvvəlki hypothesis-i dəyişirsə
  köhnə hypothesis-i təkrarlama.
- Diaqnoz qoyma.
- "Bu mütləq özgüvənsizlikdir" demə.
- Ehtimal dili istifadə et:
  "əlaqəli ola bilər",
  "bu situativ görünür",
  "tək bu məlumatla qəti demək olmaz".
- Valideynin konkret ehtiyacı ilə proqramı əlaqələndir.
- Generic bacarıqlar siyahısını səbəbsiz sadalama.
- Need aydın deyilsə maksimum 1 qısa clarification sualı.
- Need aydındırsa discovery-ni sırf tamamlamaq üçün uzatma.
- Ready lead-dirsə discovery yoxdur.
- Uşaq müqavimətində:
  səbəbi anla -> məcbur etmə ->
  uşaq razıdırsa qısa görüntülü tanışlıq.
- Eyni cavabı sözbəsöz təkrarlama.
- Hər cavabın sonunda avtomatik sual vermə.

Üslub:
- 2–4 qısa cümlə
- sadə FAQ 20–40 söz
- izahlı cavab adətən 50–70 sözdən çox olmasın
- WhatsApp/Instagram tərzi
- professional və təbii

5/9 aylıq yol:
- şəxsi/sosial bacarıqlar -> ilkin 5 aylıq Foundation + Leadership
- gələcək istiqaməti, qərarvermə, potensialın praktik nəticəsi -> 9 aylıq yol
- bu sərt keyword mapping deyil
- qarışıq ehtiyac -> 1 clarification
- yekun fit mütəxəssis tərəfindən görüntülü tanışlıqdan sonra təsdiqlənir

Yalnız final cavabı yaz.
"""


def generate_contextual_response(
    user_text,
    lead,
    analysis,
    include_next_step=True
):

    faq_hits = retrieve_similar(
        user_text,
        top_k=7
    )

    kb_context = []

    for hit in faq_hits:

        if hit[
            "score"
        ] >= 0.10:

            kb_context.append({

                "question":
                    hit["question"],

                "answer":
                    hit["answer"],

                "score":
                    round(
                        hit["score"],
                        3
                    )
            })

    child = get_active_child(
        lead
    )

    recommendation = (
        calculate_recommendation(
            lead,
            analysis
        )
    )

    prompt = f"""
CURRENT USER MESSAGE:
{user_text}

PRIMARY INTENT:
{analysis.get("primary_intent")}

ALL INTENTS:
{json_dumps(analysis.get("all_intents", []))}

QUESTIONS:
{json_dumps(analysis.get("questions", []))}

CURRENT STATE:
{json_dumps(sanitize_state_for_llm(lead))}

ACTIVE CHILD:
{json_dumps(child)}

NEED ANALYSIS:
{json_dumps(analysis.get("need_analysis", {}))}

PRELIMINARY RECOMMENDATION:
{recommendation}

RECENT CONVERSATION:
{recent_history_text(lead)}

APPROVED BUSINESS FACTS:
{json_dumps(BUSINESS_FACTS)}

RELEVANT KNOWLEDGE BASE:
{json_dumps(kb_context)}

NEXT STEP MAY BE INCLUDED:
{include_next_step}

Əvvəl primary intent-i cavablandır.
Əgər valideyn situasiya təsvir edibsə,
məlumatları bir-biri ilə əlaqələndir.

Məsələn:
evdə rahat danışır,
amma məktəbdə müəllim qarşısında
bildiyi cavabı deməyə çəkinir.

Burada:
"bu mütləq özgüvənsizlikdir"
demək olmaz.

Belə cavab daha düzgündür:
"Bu, ümumi xarakter xüsusiyyətindən çox
konkret məktəb mühitində özünüifadə və
özünəinamla əlaqəli ola bilər.
Amma tək bu məlumatla qəti nəticə demək olmaz."

State-də olan məlumatı təkrar soruşma.
Yalnız real faydası varsa maksimum bir sual ver.
"""

    try:

        response = client.chat.completions.create(

            model=MODEL_NAME,

            temperature=0.25,

            messages=[
                {
                    "role": "system",
                    "content": RESPONSE_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return (
            response
            .choices[0]
            .message
            .content
            .strip()
        )

    except Exception as exc:

        print(
            "RESPONSE LLM ERROR:",
            repr(exc)
        )

        return (
            "Bu məqamı dəqiq cavablandırmaq üçün "
            "məlumatı məsul əməkdaşla dəqiqləşdirmək daha doğru olar."
        )


# ============================================================
# 24. NEXT BEST ACTION
# ============================================================

def get_next_best_action(
    lead,
    analysis=None
):

    if human_owns_lead(
        lead
    ):
        return None

    if lead.get(
        "agreed_followup_at"
    ):
        return None

    if lead.get(
        "ready_to_proceed"
    ):

        if lead.get(
            "child_intro_status"
        ) in {
            "NOT_STARTED",
            "PENDING"
        }:

            return "BOOK_CHILD_INTRO"

        return None

    child = get_active_child(
        lead
    )

    if child.get(
        "age"
    ) is None:

        return "ASK_CHILD_AGE"

    age_rule = evaluate_age_rule(
        child
    )

    if age_rule[
        "status"
    ] in {
        "SPECIALIST_REVIEW",
        "NOT_FIT_GROUP"
    }:

        return "SPECIALIST_REVIEW"

    if age_rule[
        "status"
    ] == "INDIVIDUAL":

        return "INDIVIDUAL_OFFER"

    if not child.get(
        "need"
    ):

        return "ASK_NEED"

    if analysis:

        need_analysis = (
            analysis.get(
                "need_analysis",
                {}
            )
            or {}
        )

        if (
            not child.get(
                "discovery_complete"
            )
            and
            need_analysis.get(
                "clarification_needed"
            )
            and
            child.get(
                "discovery_question_count",
                0
            ) < 3
        ):

            return "CLARIFY_NEED"

    if (
        not lead.get("phone")
        and
        not lead.get("parent_name")
    ):

        return "ASK_NAME_PHONE"

    if not lead.get(
        "phone"
    ):

        return "ASK_PHONE"

    if not lead.get(
        "preferred_call_time"
    ):

        return "ASK_CHILD_INTRO_TIME"

    return "BOOK_CHILD_INTRO"


# ============================================================
# 25. RENDER NEXT ACTION
# ============================================================

def render_next_action(
    lead,
    analysis=None
):

    action = get_next_best_action(
        lead,
        analysis
    )

    lead[
        "next_action"
    ] = action

    child = get_active_child(
        lead
    )

    if action is None:
        return None

    if action == "ASK_CHILD_AGE":

        lead[
            "_last_question_topic"
        ] = "child_age"

        if child.get(
            "name"
        ):

            return (
                f"{child['name']}ın neçə yaşı var?"
            )

        return (
            "Övladınızın neçə yaşı var?"
        )

    if action == "ASK_NEED":

        lead[
            "_last_question_topic"
        ] = "need"

        return (
            "Övladınızda hazırda ən çox hansı tərəfin "
            "inkişaf etməsini istəyirsiniz?"
        )

    if action == "CLARIFY_NEED":

        need_analysis = (
            analysis.get(
                "need_analysis",
                {}
            )
            if analysis
            else {}
        )

        question = (
            need_analysis.get(
                "best_clarification_question"
            )
        )

        if question:

            child[
                "discovery_question_count"
            ] += 1

            lead[
                "_last_question_topic"
            ] = "consultative_discovery"

            return question

        return None

    if action == "ASK_NAME_PHONE":

        lead[
            "_last_question_topic"
        ] = "contact"

        return (
            "Adınızı və əlaqə nömrənizi yaza bilərsiniz?"
        )

    if action == "ASK_PHONE":

        lead[
            "_last_question_topic"
        ] = "phone"

        return (
            "Sizinlə əlaqə saxlaya bilməyimiz üçün "
            "telefon nömrənizi yaza bilərsiniz?"
        )

    if action == "ASK_CHILD_INTRO_TIME":

        lead[
            "_last_question_topic"
        ] = "child_intro_time"

        lead[
            "sales_stage"
        ] = "CHILD_INTRO_PENDING"

        lead[
            "child_intro_status"
        ] = "PENDING"

        return (
            "Övladınızla qısa görüntülü tanışlıq üçün "
            "sizə hansı gün və saat aralığı uyğundur?"
        )

    if action == "BOOK_CHILD_INTRO":

        if lead.get(
            "preferred_call_time"
        ):

            lead[
                "child_intro_status"
            ] = "BOOKED"

            lead[
                "sales_stage"
            ] = "CHILD_INTRO_BOOKED"

            lead[
                "status"
            ] = "CALL_REQUESTED"

            lead[
                "next_action"
            ] = "CHILD_INTRO"

            return (
                f"{lead['preferred_call_time']} üçün "
                "övladınızla qısa görüntülü tanışlıq "
                "müraciətini qeyd etdim ✅"
            )

        return (
            "Övladınızla qısa görüntülü tanışlıq üçün "
            "sizə uyğun gün və saat aralığını yaza bilərsiniz?"
        )

    if action == "SPECIALIST_REVIEW":

        lead[
            "recommended_path"
        ] = "SPECIALIST_REVIEW"

        return (
            "Bu hal üçün uyğunluğu mütəxəssisin "
            "qısa görüntülü tanışlıqda qiymətləndirməsi "
            "daha doğru olar."
        )

    if action == "INDIVIDUAL_OFFER":

        return (
            "Bu yaş üçün qrup proqramından çox fərdi coaching "
            "uyğun seçimdir. Görüş 30–45 dəqiqədir və "
            "bir görüş 80 AZN-dir."
        )

    return None


# ============================================================
# 26. SPECIAL RESPONSES
# ============================================================

def handle_child_resistance(
    lead
):

    child = get_active_child(
        lead
    )

    child[
        "willingness"
    ] = "unwilling"

    return (
        "Məcbur etmək tövsiyə olunmur. "
        "Əvvəlcə niyə istəmədiyini anlamaq daha faydalıdır. "
        "Özü razı olsa, proqramı birbaşa eşitməsi üçün "
        "təxminən 5 dəqiqəlik görüntülü tanışlıq edə bilərik."
    )


def handle_followup_commitment(
    lead
):

    when = lead.get(
        "agreed_followup_at"
    )

    lead[
        "next_action"
    ] = "AGREED_FOLLOWUP"

    if when:

        return (
            f"Əlbəttə. {when} üçün qeyd edirəm. "
            "O vaxta qədər əlavə follow-up göndərməyəcəyik."
        )

    return (
        "Əlbəttə. Sizə uyğun vaxtda qaldığımız yerdən davam edə bilərik."
    )


# ============================================================
# 27. APPEND NEXT STEP
# ============================================================

def append_next_step_if_needed(
    base_response,
    lead,
    analysis
):

    primary = analysis.get(
        "primary_intent"
    )

    if primary in {
        "state_recall",
        "followup_commitment",
        "human_request",
        "complaint",
        "clinical_risk",
        "partnership",
        "special_payment",
        "ready_to_proceed"
    }:

        return base_response

    child = get_active_child(
        lead
    )

    has_lead_context = any([
        child.get("age") is not None,
        bool(child.get("need")),
        bool(lead.get("phone")),
        bool(lead.get("parent_name"))
    ])

    if (
        primary in {
            "location",
            "schedule",
            "duration",
            "language",
            "program_info",
            "price",
            "eligibility"
        }
        and
        not has_lead_context
    ):

        return base_response

    next_text = render_next_action(
        lead,
        analysis
    )

    if not next_text:
        return base_response

    if (
        normalize_text(
            next_text
        )
        in
        normalize_text(
            base_response
        )
    ):

        return base_response

    return (
        base_response.rstrip()
        +
        "\n\n"
        +
        next_text
    )


# ============================================================
# 28. MAIN AGENT
# ============================================================

def lead_agent_reply(
    user_text,
    lead,
    faq_min_score=0.25,
    history=None,
    conversation_history=None
):

    # app.py backward compatibility
    if conversation_history is None:
        conversation_history = history

    user_text = compact_spaces(
        user_text
    )

    if not user_text:
        return "Mesajınızı yaza bilərsiniz."

    ensure_children(
        lead
    )

    # If app passed history and engine has no history yet
    if (
        conversation_history
        and
        not lead.get(
            "_conversation_history"
        )
    ):

        normalized_history = []

        for item in conversation_history[-12:]:

            if isinstance(
                item,
                dict
            ):

                role = item.get(
                    "role",
                    ""
                )

                content = item.get(
                    "content",
                    ""
                )

                if role in {
                    "user",
                    "assistant"
                }:

                    normalized_history.append({
                        "role": role,
                        "content": content
                    })

        if normalized_history:

            lead[
                "_conversation_history"
            ] = normalized_history

    # ---------------------------------------
    # HUMAN OWNERSHIP HARD STOP
    # ---------------------------------------

    if human_owns_lead(
        lead
    ):

        response = (
            "Müraciətiniz artıq əməkdaşımıza yönləndirilib. "
            "Paralel olaraq əlavə satış mesajı göndərməyəcəyəm."
        )

        add_history(
            lead,
            "user",
            user_text
        )

        add_history(
            lead,
            "assistant",
            response
        )

        return response

    # ---------------------------------------
    # ANALYZE
    # ---------------------------------------

    analysis = analyze_user_message(
        user_text,
        lead
    )

    lead[
        "_last_analysis"
    ] = analysis

    lead[
        "_last_user_message"
    ] = user_text

    print(
        "INTENT DEBUG:",
        {
            "primary_intent":
                analysis.get(
                    "primary_intent"
                ),

            "all_intents":
                analysis.get(
                    "all_intents"
                ),

            "confidence":
                analysis.get(
                    "confidence"
                )
        }
    )

    # ---------------------------------------
    # STATE UPDATE BACKGROUND
    # ---------------------------------------

    apply_analysis_to_state(
        lead,
        analysis
    )

    calculate_recommendation(
        lead,
        analysis
    )

    primary = analysis.get(
        "primary_intent",
        "other"
    )

    # ---------------------------------------
    # SAFETY
    # ---------------------------------------

    if (
        analysis.get(
            "clinical_or_safety_risk"
        )
        or primary == "clinical_risk"
    ):

        mark_handoff(
            lead
        )

        response = (
            "Bu məsələ Junior Coaching-in inkişaf proqramı "
            "çərçivəsindən kənar ola bilər. "
            "Daha düzgün qiymətləndirmə üçün müraciətinizi "
            "məsul əməkdaşa yönləndirmək daha doğru olar."
        )

    # ---------------------------------------
    # COMPLAINT
    # ---------------------------------------

    elif (
        analysis.get(
            "complaint"
        )
        or primary == "complaint"
    ):

        mark_handoff(
            lead
        )

        response = (
            "Narazılığınızı başa düşürəm. "
            "Məsələnin düzgün araşdırılması üçün "
            "müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # ---------------------------------------
    # HUMAN REQUEST
    # ---------------------------------------

    elif (
        analysis.get(
            "human_requested"
        )
        or primary == "human_request"
    ):

        mark_handoff(
            lead
        )

        if lead.get(
            "phone"
        ):

            response = (
                "Əlbəttə. Müraciətinizi əməkdaşımıza "
                "yönləndirirəm. Əlaqə nömrəniz artıq qeyd olunub."
            )

        else:

            response = (
                "Əlbəttə. Müraciətinizi əməkdaşımıza "
                "yönləndirə bilərəm. "
                "Əlaqə nömrənizi yaza bilərsiniz?"
            )

    # ---------------------------------------
    # PARTNERSHIP
    # ---------------------------------------

    elif (
        analysis.get(
            "partnership"
        )
        or primary == "partnership"
    ):

        mark_handoff(
            lead
        )

        response = (
            "Əməkdaşlıq təkliflərini aidiyyəti komanda "
            "dəyərləndirir. Müraciətinizi həmin komandaya "
            "yönləndirə bilərik."
        )

    # ---------------------------------------
    # SPECIAL PAYMENT
    # ---------------------------------------

    elif (
        analysis.get(
            "special_payment_request"
        )
        or primary == "special_payment"
    ):

        response = (
            "Standart qaydada ödəniş modul başlamazdan əvvəl "
            "modul üzrə edilir. Başlamağa əsas maneə "
            "ödəniş formasıdırsa, bunu rəhbərliklə ayrıca "
            "dəqiqləşdirmək olar."
        )

        lead[
            "next_action"
        ] = "PAYMENT_HANDOFF_IF_HIGH_INTENT"

    # ---------------------------------------
    # STATE RECALL
    # ---------------------------------------

    elif (
        primary == "state_recall"
        or
        (
            analysis.get(
                "state_recall",
                {}
            )
            or {}
        ).get(
            "requested"
        )
    ):

        recall_info = (
            analysis.get(
                "state_recall",
                {}
            )
            or {}
        )

        response = answer_state_recall(
            lead,
            recall_info.get(
                "fields",
                []
            )
        )

    # ---------------------------------------
    # READY TO PROCEED
    # ---------------------------------------

    elif (
        analysis.get(
            "ready_to_proceed"
        )
        or
        primary in {
            "ready_to_proceed",
            "registration"
        }
    ):

        response = (
            handle_ready_to_proceed(
                lead
            )
        )

    # ---------------------------------------
    # FOLLOW-UP
    # ---------------------------------------

    elif primary == "followup_commitment":

        response = (
            handle_followup_commitment(
                lead
            )
        )

    # ---------------------------------------
    # CHILD RESISTANCE
    # ---------------------------------------

    elif primary == "child_resistance":

        response = (
            handle_child_resistance(
                lead
            )
        )

        response = append_next_step_if_needed(
            response,
            lead,
            analysis
        )

    # ---------------------------------------
    # LOCATION
    # ---------------------------------------

    elif primary == "location":

        if len(
            analysis.get(
                "questions",
                []
            )
        ) > 1:

            response = (
                generate_contextual_response(
                    user_text,
                    lead,
                    analysis
                )
            )

        else:

            response = (
                answer_location()
            )

        response = (
            append_next_step_if_needed(
                response,
                lead,
                analysis
            )
        )

    # ---------------------------------------
    # SCHEDULE
    # ---------------------------------------

    elif primary == "schedule":

        if len(
            analysis.get(
                "questions",
                []
            )
        ) > 1:

            response = (
                generate_contextual_response(
                    user_text,
                    lead,
                    analysis
                )
            )

        else:

            response = (
                answer_schedule()
            )

        response = (
            append_next_step_if_needed(
                response,
                lead,
                analysis
            )
        )

    # ---------------------------------------
    # DURATION
    # ---------------------------------------

    elif primary == "duration":

        response = (
            generate_contextual_response(
                user_text,
                lead,
                analysis
            )
        )

        response = (
            append_next_step_if_needed(
                response,
                lead,
                analysis
            )
        )

    # ---------------------------------------
    # LANGUAGE
    # ---------------------------------------

    elif primary == "language":

        response = (
            answer_language()
        )

        response = (
            append_next_step_if_needed(
                response,
                lead,
                analysis
            )
        )

    # ---------------------------------------
    # ELIGIBILITY
    # ---------------------------------------

    elif primary == "eligibility":

        response = (
            answer_eligibility(
                lead
            )
        )

        response = (
            append_next_step_if_needed(
                response,
                lead,
                analysis
            )
        )

    # ---------------------------------------
    # PRICE
    # ---------------------------------------

    elif primary == "price":

        if len(
            analysis.get(
                "questions",
                []
            )
        ) > 1:

            response = (
                generate_contextual_response(
                    user_text,
                    lead,
                    analysis
                )
            )

        else:

            response = (
                answer_price(
                    lead
                )
            )

        response = (
            append_next_step_if_needed(
                response,
                lead,
                analysis
            )
        )

    # ---------------------------------------
    # GREETING
    # ---------------------------------------

    elif primary == "greeting":

        has_history = (
            len(
                lead.get(
                    "_conversation_history",
                    []
                )
            )
            > 0
        )

        if has_history:

            response = "Salam 😊"

        else:

            response = (
                "Salam 😊 Sizə necə kömək edə bilərəm?"
            )

    # ---------------------------------------
    # CONSULTATION / QUESTIONS / PROGRAM INFO
    # ---------------------------------------

    elif (
        primary in {
            "consultation",
            "program_info",
            "other"
        }
        or
        analysis.get(
            "is_question"
        )
        or
        analysis.get(
            "questions"
        )
    ):

        response = (
            generate_contextual_response(
                user_text,
                lead,
                analysis
            )
        )

        response = (
            append_next_step_if_needed(
                response,
                lead,
                analysis
            )
        )

    # ---------------------------------------
    # FIELD INFORMATION
    # ---------------------------------------

    else:

        next_text = render_next_action(
            lead,
            analysis
        )

        if next_text:

            response = next_text

        else:

            response = (
                generate_contextual_response(
                    user_text,
                    lead,
                    analysis,
                    include_next_step=False
                )
            )

    # ---------------------------------------
    # AUTO BOOKING CHECK
    # ---------------------------------------

    if (
        lead.get(
            "preferred_call_time"
        )
        and
        lead.get(
            "child_intro_status"
        )
        in {
            "PENDING",
            "NOT_STARTED"
        }
        and
        lead.get(
            "phone"
        )
    ):

        lead[
            "child_intro_status"
        ] = "BOOKED"

        lead[
            "sales_stage"
        ] = "CHILD_INTRO_BOOKED"

        lead[
            "next_action"
        ] = "CHILD_INTRO"

        lead[
            "status"
        ] = "CALL_REQUESTED"

    # ---------------------------------------
    # HISTORY
    # ---------------------------------------

    add_history(
        lead,
        "user",
        user_text
    )

    add_history(
        lead,
        "assistant",
        response
    )

    lead[
        "_last_bot_response"
    ] = response

    sync_legacy_fields(
        lead
    )

    # IMPORTANT:
    # app.py compatibility
    return response


# ============================================================
# 29. OLD FLOW COMPATIBILITY
# ============================================================

def get_next_missing_field(
    lead
):

    child = get_active_child(
        lead
    )

    if child.get(
        "age"
    ) is None:

        return "child_age"

    if not child.get(
        "need"
    ):

        return "main_concern"

    if (
        not lead.get(
            "parent_name"
        )
        and
        not lead.get(
            "phone"
        )
    ):

        return "contact"

    if not lead.get(
        "phone"
    ):

        return "phone"

    if not lead.get(
        "preferred_call_time"
    ):

        return "preferred_call_time"

    return None


FIELD_QUESTIONS = {

    "parent_name":
        "Sizə necə müraciət edə bilərəm?",

    "child_name":
        "Övladınızın adını öyrənə bilərəm?",

    "child_age":
        "Övladınızın neçə yaşı var?",

    "main_concern":
        (
            "Övladınızda hazırda ən çox hansı tərəfin "
            "inkişaf etməsini istəyirsiniz?"
        ),

    "phone":
        (
            "Sizinlə əlaqə saxlaya bilməyimiz üçün "
            "telefon nömrənizi yaza bilərsiniz?"
        ),

    "preferred_call_time":
        (
            "Övladınızla qısa görüntülü tanışlıq üçün "
            "sizə hansı gün və saat aralığı uyğundur?"
        )
}


def answer_faq_question(
    question,
    min_score=0.25
):

    hit = get_best_faq_hit(
        question,
        min_score=min_score
    )

    if not hit:

        return (
            "Bu sualla bağlı təsdiqlənmiş məlumat "
            "bazasından dəqiq cavab tapa bilmədim."
        )

    return hit[
        "answer"
    ]


# ============================================================
# 30. SAVE CONVERSATION LOG
# BACKWARD COMPATIBLE WITH OLD APP.PY
# ============================================================

def save_conversation_log(
    session_id,
    user_message,
    bot_response,
    current_field=None,
    lead=None,
    analysis=None,
    faq_score=None
):

    # compatibility:
    # save_conversation_log(
    # session_id, user, answer, lead
    # )

    if (
        isinstance(
            current_field,
            dict
        )
        and
        lead is None
    ):

        lead = current_field
        current_field = None

    if lead is None:

        raise ValueError(
            "lead məlumatı verilməyib."
        )

    sync_legacy_fields(
        lead
    )

    if analysis is None:

        analysis = (
            lead.get(
                "_last_analysis"
            )
            or {}
        )

    if faq_score is None:

        faq_score = lead.get(
            "_last_faq_score"
        )

    conn = get_connection()

    conn.execute(
        """
        INSERT INTO conversation_logs (

            session_id,

            user_message,
            bot_response,

            intent,
            confidence,

            faq_score,
            current_field,

            parent_name,
            parent_title,

            child_name,
            child_age,
            main_concern,

            phone,
            preferred_call_time,

            status,
            source,

            state_json,
            analysis_json,

            created_at
        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,

        (
            session_id,

            user_message,
            bot_response,

            analysis.get(
                "primary_intent"
            ),

            analysis.get(
                "confidence"
            ),

            faq_score,

            current_field,

            lead.get(
                "parent_name"
            ),

            lead.get(
                "parent_title"
            ),

            lead.get(
                "child_name"
            ),

            lead.get(
                "child_age"
            ),

            lead.get(
                "main_concern"
            ),

            lead.get(
                "phone"
            ),

            lead.get(
                "preferred_call_time"
            ),

            lead.get(
                "status"
            ),

            lead.get(
                "source"
            ),

            json_dumps(
                sanitize_state_for_llm(
                    lead
                )
            ),

            json_dumps(
                analysis
            ),

            now_string()
        )
    )

    conn.commit()
    conn.close()


# ============================================================
# 31. ADMIN DATA
# ============================================================

def get_all_leads():

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM leads
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


def get_all_conversation_logs():

    conn = get_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM conversation_logs
        ORDER BY id DESC
        """
    ).fetchall()

    conn.close()

    return [
        dict(row)
        for row in rows
    ]


def get_leads_dataframe():

    import pandas as pd

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM leads
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return df


def get_conversation_logs_dataframe():

    import pandas as pd

    conn = get_connection()

    df = pd.read_sql_query(
        """
        SELECT *
        FROM conversation_logs
        ORDER BY id DESC
        """,
        conn
    )

    conn.close()

    return df


# ============================================================
# END
# ============================================================