# ============================================================
# JUNIOR COACHING
# AI SALES & CONVERSATION ENGINE
# V10.2
#
# Understand
# -> Extract
# -> Reason
# -> Build Response Obligations
# -> Answer
# -> Update State
# -> Policy Based Next Step
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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FAQ_PATH = os.path.join(
    BASE_DIR,
    "Junior_Coaching_sesli_AI_FAQ.txt"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "junior_coaching.db"
)

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

MODEL_NAME = os.getenv(
    "OPENAI_MODEL",
    "gpt-4o-mini"
)


# ============================================================
# 2. OPENAI CLIENT
# ============================================================

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

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
# 3. APPROVED BUSINESS FACTS
#
# IMPORTANT:
# LLM hard-fact yaratmamalıdır.
# Bu bölmə source-of-truth rolunu oynayır.
# ============================================================

BUSINESS_FACTS = {

    "program_name":
        "Junior Coaching",

    "age_min":
        12,

    "age_max":
        18,

    "near_age_rule":
        (
            "12 yaşa çox yaxın olan uşaqlar avtomatik "
            "rədd edilmir. Uyğunluğu mütəxəssis ayrıca "
            "qiymətləndirə bilər."
        ),

    "format":
        (
            "Canlı qrup formatıdır. Praktik məşqlər, "
            "komanda işi, situasiyalar, layihələr və "
            "təqdimatlardan istifadə olunur."
        ),

    "group_frequency":
        "Ayda 3 bazar günü",

    "group_session_duration":
        "2 saat",

    "full_program_duration":
        "9 ay / 27 görüş",

    "language":
        (
            "Görüşlər əsasən Azərbaycan dilində keçirilir. "
            "Ehtiyac olduqda bəzi materiallar rus və ya "
            "ingilis dilində təqdim oluna bilər."
        ),

    "address":
        (
            "Süleyman Sani Axundov küçəsi, "
            "ADAS Plaza — ELİT T/M yaxınlığı"
        ),

    "parent_initial_call_duration":
        "5–7 dəqiqə",

    "child_intro_duration":
        "təxminən 5 dəqiqə",

    "individual_coaching_duration":
        "30–45 dəqiqə",

    "individual_coaching_price":
        80,

    # Group module prices are NOT approved yet
    "foundation_price":
        None,

    "leadership_price":
        None,

    "pro_price":
        None,

    "impact_price":
        None,

    "therapy_boundary":
        (
            "Junior Coaching terapiya, psixoloji diaqnostika "
            "və ya psixiatrik müalicə xidməti deyil."
        )
}


# ============================================================
# 4. ENUMS / CONSTANTS
# ============================================================

CHILD_WILLINGNESS_VALUES = {
    "unknown",
    "willing",
    "hesitant",
    "unwilling"
}


# ============================================================
# 5. BASIC HELPERS
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

    text = safe_text(
        text
    ).lower()

    replacements = {
        "ə": "e",
        "ı": "i",
        "ö": "o",
        "ü": "u",
        "ş": "s",
        "ç": "c",
        "ğ": "g"
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new
        )

    return re.sub(
        r"\s+",
        " ",
        text
    ).strip()


def normalize_phone(value):

    if not value:
        return None

    digits = re.sub(
        r"\D",
        "",
        str(value)
    )

    if (
        len(digits) == 10
        and digits.startswith("0")
    ):
        return digits

    if (
        len(digits) == 12
        and digits.startswith("994")
    ):
        return digits

    return None


def safe_int(value):

    if value is None:
        return None

    if isinstance(
        value,
        int
    ):
        return value

    match = re.search(
        r"\b(\d{1,2})\b",
        str(value)
    )

    if not match:
        return None

    try:
        return int(
            match.group(1)
        )
    except Exception:
        return None


def json_dumps(data):

    return json.dumps(
        data,
        ensure_ascii=False,
        default=str
    )


# ============================================================
# 6. CHILD STATE
# ============================================================

def create_empty_child():

    return {

        "child_id":
            str(uuid.uuid4()),

        "name":
            None,

        "age":
            None,

        "need":
            None,

        "need_tags":
            [],

        "context":
            None,

        "duration":
            None,

        "impact":
            None,

        "desired_outcome":
            None,

        "willingness":
            "unknown",

        "hypothesis":
            None,

        "hypothesis_confidence":
            None,

        "recommendation_signal":
            "unknown",

        "recommended_path":
            None,

        "discovery_question_count":
            0,

        "discovery_complete":
            False
    }


# ============================================================
# 7. LEAD STATE
# ============================================================

def create_empty_lead(
    source="Unknown"
):

    child = create_empty_child()

    return {

        # ----------------------------------------------------
        # PARENT
        # ----------------------------------------------------

        "parent_name":
            None,

        "parent_title":
            None,

        "phone":
            None,

        # ----------------------------------------------------
        # MULTI CHILD
        # ----------------------------------------------------

        "children":
            [child],

        "active_child_index":
            0,

        # ----------------------------------------------------
        # LEGACY APP COMPATIBILITY
        # ----------------------------------------------------

        "child_name":
            None,

        "child_age":
            None,

        "main_concern":
            None,

        "needs_concern_followup":
            False,

        "concern_duration":
            None,

        "concern_onset":
            None,

        # ----------------------------------------------------
        # SALES / BUSINESS FLOW
        # ----------------------------------------------------

        "sales_stage":
            "NEW",

        "ready_to_proceed":
            False,

        # Normal sales path:
        # lead -> parent call -> optional child intro

        "parent_call_status":
            "NOT_STARTED",

        # NOT_STARTED / PENDING / BOOKED / COMPLETED

        "parent_call_time":
            None,

        "child_intro_status":
            "NOT_STARTED",

        # NOT_STARTED / PENDING / BOOKED / COMPLETED

        "child_intro_time":
            None,

        "child_intro_required":
            False,

        "fit_status":
            "UNKNOWN",

        "payment_status":
            "NOT_STARTED",

        "recommended_path":
            None,

        # ----------------------------------------------------
        # OBJECTION / DECISION
        # ----------------------------------------------------

        "objection_type":
            None,

        "decision_blocker":
            None,

        "ambiguity_present":
            False,

        "clarification_needed":
            False,

        # ----------------------------------------------------
        # FOLLOWUP
        # ----------------------------------------------------

        "preferred_call_time":
            None,

        "agreed_followup_at":
            None,

        # ----------------------------------------------------
        # HUMAN OWNERSHIP
        # ----------------------------------------------------

        "handoff_status":
            "none",

        "owner":
            "AI",

        # ----------------------------------------------------
        # ORCHESTRATION
        # ----------------------------------------------------

        "primary_intent":
            None,

        "all_intents":
            [],

        "response_obligations":
            [],

        "next_action":
            None,

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        "_conversation_history":
            [],

        "_last_analysis":
            None,

        "_last_bot_response":
            None,

        "_last_user_message":
            None,

        "_last_question_topic":
            None,

        "_last_intent":
            None,

        "_last_confidence":
            None,

        "_last_faq_score":
            None,

        # ----------------------------------------------------
        # LEGACY
        # ----------------------------------------------------

        "source":
            source,

        "status":
            "NEW"
    }


# ============================================================
# 8. CHILD HELPERS
# ============================================================

def ensure_children(lead):

    if "children" not in lead:

        lead["children"] = []

    if not lead["children"]:

        lead["children"].append(
            create_empty_child()
        )

    if lead.get(
        "active_child_index"
    ) is None:

        lead["active_child_index"] = 0

    if (
        lead["active_child_index"]
        >= len(lead["children"])
    ):

        lead["active_child_index"] = 0

    return lead["children"]


def get_active_child(lead):

    ensure_children(
        lead
    )

    return lead[
        "children"
    ][
        lead.get(
            "active_child_index",
            0
        )
    ]


def sync_legacy_fields(lead):

    child = get_active_child(
        lead
    )

    lead["child_name"] = (
        child.get("name")
    )

    lead["child_age"] = (
        child.get("age")
    )

    lead["main_concern"] = (
        child.get("need")
    )

    lead["concern_duration"] = (
        child.get("duration")
    )

    lead["concern_onset"] = (
        child.get("context")
    )

    lead["recommended_path"] = (
        child.get(
            "recommended_path"
        )
    )

    # old app used preferred_call_time
    # for current booking time.
    if (
        lead.get("parent_call_time")
        and
        not lead.get(
            "preferred_call_time"
        )
    ):

        lead[
            "preferred_call_time"
        ] = lead[
            "parent_call_time"
        ]

    return lead


def find_child_by_name(
    lead,
    name
):

    if not name:
        return None

    target = normalize_text(
        name
    )

    for index, child in enumerate(
        lead.get(
            "children",
            []
        )
    ):

        if (
            normalize_text(
                child.get("name")
            )
            == target
        ):

            return index

    return None


def create_new_child(
    lead
):

    child = create_empty_child()

    lead[
        "children"
    ].append(
        child
    )

    index = (
        len(
            lead["children"]
        )
        - 1
    )

    lead[
        "active_child_index"
    ] = index

    sync_legacy_fields(
        lead
    )

    return index


# ============================================================
# 9. FAQ / KNOWLEDGE BASE
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
    ) as file:

        text = file.read()

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

    result = []

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

            result.append({
                "question":
                    question,

                "answer":
                    answer
            })

    return result


def build_faq_index():

    global FAQ_ITEMS
    global FAQ_VECTORIZER
    global FAQ_MATRIX

    FAQ_ITEMS = (
        parse_faq_file()
    )

    if not FAQ_ITEMS:

        FAQ_VECTORIZER = None
        FAQ_MATRIX = None

        return

    questions = [
        item["question"]
        for item in FAQ_ITEMS
    ]

    FAQ_VECTORIZER = (
        TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=50000
        )
    )

    FAQ_MATRIX = (
        FAQ_VECTORIZER
        .fit_transform(
            questions
        )
    )


def retrieve_similar(
    query,
    top_k=5
):

    if (
        FAQ_VECTORIZER is None
        or
        FAQ_MATRIX is None
        or
        not FAQ_ITEMS
    ):

        return []

    query_vector = (
        FAQ_VECTORIZER
        .transform(
            [query]
        )
    )

    scores = cosine_similarity(
        query_vector,
        FAQ_MATRIX
    )[0]

    indices = (
        scores
        .argsort()[::-1]
    )

    result = []

    for index in indices[:top_k]:

        result.append({

            "question":
                FAQ_ITEMS[index][
                    "question"
                ],

            "answer":
                FAQ_ITEMS[index][
                    "answer"
                ],

            "score":
                float(
                    scores[index]
                )
        })

    return result


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

    if (
        hits[0]["score"]
        < min_score
    ):

        return None

    return hits[0]


build_faq_index()


# ============================================================
# 10. DATABASE
# ============================================================

def get_connection():

    connection = sqlite3.connect(
        DB_PATH,
        check_same_thread=False
    )

    connection.row_factory = (
        sqlite3.Row
    )

    return connection


def add_column_if_missing(
    conn,
    table_name,
    column_name,
    definition
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
            {definition}
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

    new_columns = {

        "children_json":
            "TEXT",

        "sales_stage":
            "TEXT",

        "ready_to_proceed":
            "INTEGER DEFAULT 0",

        "parent_call_status":
            "TEXT",

        "parent_call_time":
            "TEXT",

        "child_intro_status":
            "TEXT",

        "child_intro_time":
            "TEXT",

        "child_intro_required":
            "INTEGER DEFAULT 0",

        "fit_status":
            "TEXT",

        "payment_status":
            "TEXT",

        "recommended_path":
            "TEXT",

        "objection_type":
            "TEXT",

        "decision_blocker":
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

    for column, definition in (
        new_columns.items()
    ):

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
# 11. DATABASE CRUD
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

    if lead.get(
        "parent_name"
    ):

        parts.append(
            "Valideyn: "
            +
            lead["parent_name"]
        )

    if lead.get(
        "phone"
    ):

        parts.append(
            "Telefon: "
            +
            lead["phone"]
        )

    for index, child in enumerate(
        lead.get(
            "children",
            []
        ),
        start=1
    ):

        values = []

        if child.get("name"):

            values.append(
                "ad="
                +
                child["name"]
            )

        if (
            child.get("age")
            is not None
        ):

            values.append(
                "yaş="
                +
                str(child["age"])
            )

        if child.get("need"):

            values.append(
                "ehtiyac="
                +
                child["need"]
            )

        if child.get(
            "desired_outcome"
        ):

            values.append(
                "nəticə="
                +
                child[
                    "desired_outcome"
                ]
            )

        if values:

            parts.append(
                f"Uşaq {index}: "
                +
                ", ".join(values)
            )

    if lead.get(
        "objection_type"
    ):

        parts.append(
            "Etiraz: "
            +
            str(
                lead[
                    "objection_type"
                ]
            )
        )

    if lead.get(
        "recommended_path"
    ):

        parts.append(
            "İlkin yol: "
            +
            str(
                lead[
                    "recommended_path"
                ]
            )
        )

    if lead.get(
        "next_action"
    ):

        parts.append(
            "Növbəti addım: "
            +
            str(
                lead[
                    "next_action"
                ]
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

    timestamp = now_string()

    conn = get_connection()

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

            parent_call_status,
            parent_call_time,

            child_intro_status,
            child_intro_time,
            child_intro_required,

            fit_status,
            payment_status,

            recommended_path,

            objection_type,
            decision_blocker,

            handoff_status,
            owner,

            agreed_followup_at,
            next_action,

            conversation_summary
        )

        VALUES (
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
        )
        """,

        (

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
                "source"
            ),

            lead.get(
                "status"
            ),

            timestamp,
            timestamp,

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
                "parent_call_status"
            ),

            lead.get(
                "parent_call_time"
            ),

            lead.get(
                "child_intro_status"
            ),

            lead.get(
                "child_intro_time"
            ),

            int(
                bool(
                    lead.get(
                        "child_intro_required"
                    )
                )
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
                "objection_type"
            ),

            lead.get(
                "decision_blocker"
            ),

            lead.get(
                "handoff_status"
            ),

            lead.get(
                "owner"
            ),

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

    lead_id = (
        cursor.lastrowid
    )

    conn.commit()
    conn.close()

    return lead_id


# ============================================================
# 12. STATE SANITIZER
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

        "parent_call_status",
        "parent_call_time",

        "child_intro_status",
        "child_intro_time",
        "child_intro_required",

        "fit_status",
        "payment_status",

        "recommended_path",

        "objection_type",
        "decision_blocker",

        "preferred_call_time",
        "agreed_followup_at",

        "handoff_status",
        "owner",

        "primary_intent",
        "all_intents",

        "next_action"
    ]

    return {
        key: deepcopy(
            lead.get(key)
        )
        for key in keys
    }


# ============================================================
# 13. HISTORY
# ============================================================

def add_history(
    lead,
    role,
    content
):

    lead.setdefault(
        "_conversation_history",
        []
    )

    lead[
        "_conversation_history"
    ].append({

        "role":
            role,

        "content":
            content
    })

    lead[
        "_conversation_history"
    ] = lead[
        "_conversation_history"
    ][-16:]


def recent_history_text(
    lead,
    limit=10
):

    history = lead.get(
        "_conversation_history",
        []
    )[-limit:]

    result = []

    for item in history:

        label = (
            "Valideyn"
            if item.get("role") == "user"
            else "Leyla"
        )

        result.append(
            f"{label}: "
            f"{item.get('content', '')}"
        )

    return "\n".join(
        result
    )


# ============================================================
# 14. JSON LLM
# ============================================================

def call_json_llm(
    system_prompt,
    user_prompt,
    temperature=0
):

    try:

        response = (
            client
            .chat.completions
            .create(

                model=MODEL_NAME,

                temperature=temperature,

                response_format={
                    "type":
                        "json_object"
                },

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            system_prompt
                    },
                    {
                        "role":
                            "user",

                        "content":
                            user_prompt
                    }
                ]
            )
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
# 15. UNDERSTANDING / ANALYSIS
# ============================================================

ANALYSIS_SYSTEM_PROMPT = """
Sən Junior Coaching üçün conversation understanding modulusan.

İstifadəçiyə cavab yazma.
Yalnız strukturlaşdırılmış JSON qaytar.

Əsas prinsip:

UNDERSTAND
-> EXTRACT
-> REASON
-> RESPONSE OBLIGATIONS
-> NEXT STEP SIGNALS

PRIORITET:
1. Safety
2. Primary intent
3. Questions that require answers
4. Corrections
5. Explicit facts
6. Current state
7. Objections / ambiguity
8. Need reasoning
9. Sales readiness

MÜTLƏQ QAYDALAR:

1. Bir mesajda bir neçə məlumat varsa hamısını çıxar.

2. Bir mesajda bir neçə sual varsa hamısını ayrıca questions
və response_obligations daxilində göstər.

3. State update primary intent-i əvəz etməsin.

Misal:
"Qızım 11 yaşındadır, 3 aya 12 olacaq. İndi başlaya bilər?"

age=11 çıxar,
amma primary_intent=eligibility olmalıdır.

4. Adı təxmin etmə.

"bu nömrə ilə əlaqə saxlayın"
"anasıyam"
"mənə"
"maraqlanıram"

ad deyil.

5. Correction:

"Tunar yox, Turandır"
"16 yox, 15 yaşı var"

köhnə dəyəri overwrite etməlidir.

6. İkinci uşaq yalnız explicit signal ilə yaradılsın:

"digər oğlum"
"kiçik qızım"
"ikinci uşağım"

Amma "bir oğlum var", "ikinci uşaq yoxdur"
-> yeni child YARATMA.

7. CHILD RESISTANCE və SPOUSE SKEPTICISM AYRIDIR.

Child resistance:
"uşağım gəlmək istəmir"
"oğlum istəmir"
"məcbur etmək istəmirəm"

Spouse skepticism:
"yoldaşım belə proqramlara inanmır"
"atası deyir bunlar lazım deyil"
"həyat yoldaşım faydasını görmür"

8. Decision dependency:
"yoldaşımla danışmadan qərar verə bilmərəm"

Bu avtomatik follow-up commitment deyil.
Əvvəl blocker-in nə olduğunu anlamaq lazım ola bilər.

9. Ambiguity:

"gəlmək istəyirik amma alınmaya bilər"
"bilmirəm alınar ya yox"

Səbəb aydın deyilsə:
ambiguity_present=true
clarification_needed=true

Ready-to-proceed etmə.

10. Ready-to-proceed yalnız aydın high-intent olduqda:

"başlamaq istəyirik"
"qeydiyyata keçək"
"hər şey aydındır, davam edək"
"bizə uyğundur, başlayaq"

Amma:
"maraqlıdır, amma..."
"yəqin gələcəyik"
"alınmaya bilər"
-> ready deyil.

11. Need inference:

Valideyn keyword deməsə belə davranış təsvirindən
ehtiyacı strukturlaşdıra bilərsən.

Amma diaqnoz qoyma.

12. New evidence əvvəlki hypothesis ilə ziddirsə
contradicts_previous_hypothesis=true.

13. Need aydındırsa sırf discovery üçün əlavə sual istəmə.

14. Ambiguous user expression varsa yanlış FAQ seçməkdənsə
clarification ver.

15. Primary intent və all_intents ayrı saxlanmalıdır.

16. Hard-fact question topics:

location
schedule
group_session_duration
parent_call_duration
child_intro_duration
program_duration
age_range
price
payment_model
language
format

17. State recall ayrıca intent-dir.

18. Response obligation — istifadəçiyə görünən cavabda
mütləq nəzərə alınmalı elementdir.

Məsələn:

"Adım Günaydır, oğlum 14 yox 15 yaşındadır,
055..., görüşlər harada və hansı gün olur?"

obligations:
- answer location
- answer schedule

Ad, age correction və phone state-də saxlanır,
amma bunlar cavabın yerini tutmur.

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

    faq_context = [

        {
            "question":
                item["question"],

            "answer":
                item["answer"],

            "score":
                round(
                    item["score"],
                    3
                )
        }

        for item in faq_hits
    ]

    prompt = f"""
CURRENT DATE:
{get_baku_time().strftime("%Y-%m-%d")}

CURRENT STATE:
{json_dumps(sanitize_state_for_llm(lead))}

RECENT CONVERSATION:
{recent_history_text(lead)}

CURRENT USER MESSAGE:
{user_text}

RELEVANT KB CANDIDATES:
{json_dumps(faq_context)}

Aşağıdakı strukturu qaytar:

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

    "response_obligations": [
        {{
            "type": "answer_question|handle_objection|clarify|state_recall|acknowledge",
            "topic": "...",
            "priority": 1
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
            "explicit_new_child": false,
            "name": null,
            "age": null,
            "need": null,
            "need_tags": [],
            "context": null,
            "duration": null,
            "impact": null,
            "desired_outcome": null,
            "willingness": null
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

    "objection": {{
        "present": false,
        "type": "child_resistance|spouse_skepticism|decision_dependency|price|schedule|logistics|trust|value|other|none",
        "summary": null,
        "clarification_needed": false,
        "best_clarification_question": null
    }},

    "ambiguity": {{
        "present": false,
        "summary": null,
        "clarification_needed": false,
        "best_clarification_question": null
    }},

    "ready_to_proceed": false,

    "parent_call_completed": false,

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

Primary intent nümunələri:

greeting
program_info
eligibility
price
location
schedule
duration
language
consultation
provide_information
child_resistance
spouse_skepticism
decision_dependency
ambiguous_objection
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

Əgər istifadəçi:

"Yoldaşım belə proqramlara skeptik yanaşır.
Deyir uşaq bunları böyüdükcə onsuz da öyrənəcək."

primary_intent = spouse_skepticism

child_resistance YOX.

Əgər:

"Proqram maraqlıdır, gəlmək də istəyirik,
amma alınmaya bilər."

ready_to_proceed=false
primary_intent=ambiguous_objection
ambiguity.present=true.

Əgər:

"Yoldaşımla danışmadan qərar verə bilmərəm."

primary_intent=decision_dependency

Əgər concrete followup date də deyirsə:
"cümə günü cavab verəcəyəm"

agreed_followup_at çıxar.

Diaqnoz qoyma.
"""

    result = call_json_llm(
        ANALYSIS_SYSTEM_PROMPT,
        prompt,
        temperature=0
    )

    if result:
        return result

    return fallback_analysis(
        user_text
    )


# ============================================================
# 16. FALLBACK ANALYSIS
# ============================================================

def fallback_analysis(
    user_text
):

    text = normalize_text(
        user_text
    )

    primary = (
        "provide_information"
    )

    if text in {
        "salam",
        "slm",
        "salamlar"
    }:

        primary = "greeting"

    elif any(
        word in text
        for word in [
            "unvan",
            "harada",
            "adres",
            "mekan"
        ]
    ):

        primary = "location"

    elif any(
        word in text
        for word in [
            "qiymet",
            "ne qeder"
        ]
    ):

        primary = "price"

    return {

        "primary_intent":
            primary,

        "all_intents":
            [primary],

        "confidence":
            0.35,

        "is_question":
            "?" in user_text,

        "questions":
            [],

        "response_obligations":
            [],

        "parent": {
            "name": None,
            "title": None,
            "phone": None
        },

        "children":
            [],

        "corrections":
            [],

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
                0,

            "contradicts_previous_hypothesis":
                False,

            "clarification_needed":
                False,

            "best_clarification_question":
                None,

            "recommendation_signal":
                "unknown"
        },

        "objection": {

            "present":
                False,

            "type":
                "none",

            "summary":
                None,

            "clarification_needed":
                False,

            "best_clarification_question":
                None
        },

        "ambiguity": {

            "present":
                False,

            "summary":
                None,

            "clarification_needed":
                False,

            "best_clarification_question":
                None
        },

        "ready_to_proceed":
            False,

        "parent_call_completed":
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
# 17. APPLY CORRECTIONS
# ============================================================

def apply_corrections(
    lead,
    analysis
):

    for correction in (
        analysis.get(
            "corrections",
            []
        )
        or []
    ):

        field = correction.get(
            "field"
        )

        new_value = correction.get(
            "new_value"
        )

        reference = correction.get(
            "child_reference"
        )

        if (
            field == "parent_name"
            and new_value
        ):

            lead[
                "parent_name"
            ] = compact_spaces(
                new_value
            )

            continue

        if field == "phone":

            phone = normalize_phone(
                new_value
            )

            if phone:
                lead["phone"] = phone

            continue

        if field not in {
            "child_name",
            "child_age",
            "need"
        }:

            continue

        child_index = None

        if reference:

            child_index = (
                find_child_by_name(
                    lead,
                    reference
                )
            )

        if child_index is None:

            child_index = (
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
        ][child_index]

        if (
            field == "child_name"
            and new_value
        ):

            child["name"] = (
                compact_spaces(
                    new_value
                )
            )

        elif field == "child_age":

            age = safe_int(
                new_value
            )

            if age is not None:

                child[
                    "age"
                ] = age

        elif (
            field == "need"
            and new_value
        ):

            child["need"] = (
                compact_spaces(
                    new_value
                )
            )

            # New need invalidates old reasoning
            child[
                "hypothesis"
            ] = None

            child[
                "hypothesis_confidence"
            ] = None

            child[
                "recommended_path"
            ] = None

            child[
                "discovery_complete"
            ] = False

    sync_legacy_fields(
        lead
    )


# ============================================================
# 18. APPLY PARENT FACTS
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

    name = parent.get(
        "name"
    )

    if name:

        invalid_names = {
            "men",
            "mene",
            "ana",
            "anasi",
            "anasiyam",
            "bu nomre",
            "nomre",
            "maraqlaniram",
            "elaqe"
        }

        if (
            normalize_text(name)
            not in invalid_names
        ):

            lead[
                "parent_name"
            ] = compact_spaces(
                name
            )

    title = normalize_text(
        parent.get(
            "title"
        )
    )

    if title == "bey":

        lead[
            "parent_title"
        ] = "bəy"

    elif title == "xanim":

        lead[
            "parent_title"
        ] = "xanım"

    phone = normalize_phone(
        parent.get(
            "phone"
        )
    )

    if phone:

        lead["phone"] = phone


# ============================================================
# 19. APPLY CHILD FACTS
# ============================================================

def apply_children_facts(
    lead,
    analysis
):

    extracted = (
        analysis.get(
            "children",
            []
        )
        or []
    )

    ensure_children(
        lead
    )

    for incoming in extracted:

        name = incoming.get(
            "name"
        )

        target = incoming.get(
            "target",
            "active"
        )

        explicit_new = bool(
            incoming.get(
                "explicit_new_child"
            )
        )

        index = None

        if name:

            index = (
                find_child_by_name(
                    lead,
                    name
                )
            )

        if (
            index is None
            and
            (
                explicit_new
                or
                target == "new"
            )
        ):

            index = create_new_child(
                lead
            )

        if index is None:

            index = lead.get(
                "active_child_index",
                0
            )

        child = lead[
            "children"
        ][index]

        if name:

            child["name"] = (
                compact_spaces(
                    name
                )
            )

        age = safe_int(
            incoming.get(
                "age"
            )
        )

        if age is not None:

            child["age"] = age

        need = incoming.get(
            "need"
        )

        if need:

            child["need"] = (
                compact_spaces(
                    need
                )
            )

        tags = (
            incoming.get(
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

            value = incoming.get(
                field
            )

            if value:

                child[field] = (
                    compact_spaces(
                        value
                    )
                )

        willingness = incoming.get(
            "willingness"
        )

        if (
            willingness
            in
            CHILD_WILLINGNESS_VALUES
        ):

            child[
                "willingness"
            ] = willingness

        lead[
            "active_child_index"
        ] = index

    sync_legacy_fields(
        lead
    )


# ============================================================
# 20. APPLY REASONING
# ============================================================

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

    if (
        need_analysis.get(
            "contradicts_previous_hypothesis"
        )
    ):

        child[
            "hypothesis"
        ] = None

        child[
            "hypothesis_confidence"
        ] = None

        child[
            "recommended_path"
        ] = None

        child[
            "discovery_complete"
        ] = False

    need_summary = (
        need_analysis.get(
            "need_summary"
        )
    )

    if need_summary:

        # Updated contextual need may replace previous simplistic need
        child["need"] = (
            compact_spaces(
                need_summary
            )
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

    signal = (
        need_analysis.get(
            "recommendation_signal",
            "unknown"
        )
    )

    child[
        "recommendation_signal"
    ] = signal

    if need_analysis.get(
        "need_is_clear"
    ):

        child[
            "discovery_complete"
        ] = True

    sync_legacy_fields(
        lead
    )


# ============================================================
# 21. APPLY ORCHESTRATION SIGNALS
# ============================================================

def apply_orchestration_signals(
    lead,
    analysis
):

    primary = analysis.get(
        "primary_intent"
    )

    lead[
        "primary_intent"
    ] = primary

    lead[
        "_last_intent"
    ] = primary

    lead[
        "_last_confidence"
    ] = analysis.get(
        "confidence"
    )

    lead[
        "all_intents"
    ] = (
        analysis.get(
            "all_intents",
            []
        )
        or []
    )

    lead[
        "response_obligations"
    ] = (
        analysis.get(
            "response_obligations",
            []
        )
        or []
    )

    objection = (
        analysis.get(
            "objection",
            {}
        )
        or {}
    )

    if objection.get(
        "present"
    ):

        lead[
            "objection_type"
        ] = objection.get(
            "type"
        )

        lead[
            "decision_blocker"
        ] = objection.get(
            "summary"
        )

    ambiguity = (
        analysis.get(
            "ambiguity",
            {}
        )
        or {}
    )

    lead[
        "ambiguity_present"
    ] = bool(
        ambiguity.get(
            "present"
        )
    )

    lead[
        "clarification_needed"
    ] = bool(
        ambiguity.get(
            "clarification_needed"
        )
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

    if analysis.get(
        "parent_call_completed"
    ):

        lead[
            "parent_call_status"
        ] = "COMPLETED"

    followup = analysis.get(
        "agreed_followup_at"
    )

    if followup:

        lead[
            "agreed_followup_at"
        ] = compact_spaces(
            followup
        )

    preferred_time = (
        analysis.get(
            "preferred_contact_time"
        )
    )

    if preferred_time:

        # IMPORTANT:
        # By default this is parent call time,
        # unless we're explicitly booking child intro.
        if (
            lead.get(
                "ready_to_proceed"
            )
            and
            lead.get(
                "child_intro_status"
            )
            in {
                "PENDING",
                "NOT_STARTED"
            }
        ):

            lead[
                "child_intro_time"
            ] = compact_spaces(
                preferred_time
            )

        else:

            lead[
                "parent_call_time"
            ] = compact_spaces(
                preferred_time
            )

            lead[
                "preferred_call_time"
            ] = compact_spaces(
                preferred_time
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

    apply_orchestration_signals(
        lead,
        analysis
    )

    sync_legacy_fields(
        lead
    )


# ============================================================
# 22. AGE RULE
# ============================================================

def evaluate_age_rule(
    child
):

    age = child.get(
        "age"
    )

    if age is None:

        return "UNKNOWN"

    if 12 <= age <= 18:

        return "GROUP_RANGE"

    if age == 11:

        return (
            "SPECIALIST_REVIEW"
        )

    if age >= 19:

        return "INDIVIDUAL"

    return "NOT_STANDARD_GROUP"


# ============================================================
# 23. PRELIMINARY RECOMMENDATION
#
# This is NOT a hard business fact.
# It is a preliminary reasoning result.
# ============================================================

def calculate_recommendation(
    lead
):

    child = get_active_child(
        lead
    )

    age_rule = evaluate_age_rule(
        child
    )

    if age_rule == "INDIVIDUAL":

        recommendation = (
            "INDIVIDUAL"
        )

    elif age_rule in {
        "SPECIALIST_REVIEW",
        "NOT_STANDARD_GROUP"
    }:

        recommendation = (
            "SPECIALIST_REVIEW"
        )

    else:

        signal = child.get(
            "recommendation_signal"
        )

        if signal == "personal_social":

            recommendation = (
                "FOUNDATION_LEADERSHIP"
            )

        elif signal == "future_direction":

            recommendation = (
                "FULL_PATH"
            )

        else:

            recommendation = None

    child[
        "recommended_path"
    ] = recommendation

    lead[
        "recommended_path"
    ] = recommendation

    return recommendation


# ============================================================
# 24. HUMAN OWNERSHIP
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
    lead
):

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
# 25. STATE RECALL
# ============================================================

def answer_state_recall(
    lead,
    fields=None
):

    fields = fields or []

    child = get_active_child(
        lead
    )

    normalized_fields = {
        normalize_text(
            field
        )
        for field in fields
    }

    parts = []

    if not normalized_fields:

        if lead.get(
            "parent_name"
        ):

            parts.append(
                f"Adınızı {lead['parent_name']} kimi qeyd etmişəm."
            )

        if child.get(
            "name"
        ):

            parts.append(
                f"Övladınızın adı {child['name']}-dır."
            )

        if (
            child.get("age")
            is not None
        ):

            parts.append(
                f"Yaşı {child['age']} olaraq qeyd olunub."
            )

        if child.get(
            "need"
        ):

            parts.append(
                f"Əsas ehtiyac kimi “{child['need']}” qeyd olunub."
            )

        return (
            " ".join(parts)
            if parts
            else
            "Bu məlumatlar hələ tam qeyd olunmayıb."
        )

    if any(
        value in normalized_fields
        for value in [
            "parent_name",
            "ad",
            "valideyn adi"
        ]
    ):

        if lead.get(
            "parent_name"
        ):

            parts.append(
                f"Adınızı {lead['parent_name']} kimi qeyd etmişəm."
            )

        else:

            parts.append(
                "Adınız hələ qeyd olunmayıb."
            )

    if any(
        value in normalized_fields
        for value in [
            "child_name",
            "usaq adi",
            "ovlad adi"
        ]
    ):

        if child.get(
            "name"
        ):

            parts.append(
                f"Övladınızın adı {child['name']}-dır."
            )

        else:

            parts.append(
                "Övladınızın adı hələ qeyd olunmayıb."
            )

    if any(
        value in normalized_fields
        for value in [
            "age",
            "yas",
            "child_age"
        ]
    ):

        if (
            child.get("age")
            is not None
        ):

            parts.append(
                f"Yaşını {child['age']} olaraq qeyd etmişəm."
            )

        else:

            parts.append(
                "Yaşı hələ qeyd olunmayıb."
            )

    if any(
        value in normalized_fields
        for value in [
            "need",
            "ehtiyac",
            "main_concern",
            "narahatliq"
        ]
    ):

        if child.get(
            "need"
        ):

            parts.append(
                f"Əsas ehtiyac kimi “{child['need']}” qeyd olunub."
            )

        else:

            parts.append(
                "Əsas ehtiyac hələ qeyd olunmayıb."
            )

    if any(
        value in normalized_fields
        for value in [
            "child_count",
            "usaq sayi",
            "ovlad sayi"
        ]
    ):

        known_children = [

            item

            for item in lead.get(
                "children",
                []
            )

            if (
                item.get("name")
                or
                item.get("age") is not None
                or
                item.get("need")
            )
        ]

        parts.append(
            f"Hazırda {len(known_children)} övlad üzrə məlumat qeyd olunub."
        )

    return " ".join(
        parts
    )


# ============================================================
# 26. HARD FACT ANSWERS
#
# These do NOT rely on LLM generation.
# ============================================================

def hard_fact_answer(
    topic,
    lead=None
):

    if topic == "location":

        return (
            "Görüşlər "
            + BUSINESS_FACTS[
                "address"
            ]
            + " ünvanında keçirilir."
        )

    if topic == "schedule":

        return (
            "Qrup görüşləri "
            + BUSINESS_FACTS[
                "group_frequency"
            ]
            + " keçirilir. Dəqiq tarix və saatlar "
              "əvvəlcədən paylaşılır."
        )

    if topic == "group_session_duration":

        return (
            "Bir qrup görüşü "
            + BUSINESS_FACTS[
                "group_session_duration"
            ]
            + " davam edir."
        )

    if topic == "parent_call_duration":

        return (
            "Valideynlə ilkin telefon danışığı adətən "
            + BUSINESS_FACTS[
                "parent_initial_call_duration"
            ]
            + " davam edir."
        )

    if topic == "child_intro_duration":

        return (
            "Övladla görüntülü tanışlıq adətən "
            + BUSINESS_FACTS[
                "child_intro_duration"
            ]
            + " davam edir."
        )

    if topic == "program_duration":

        return (
            "Tam proqram "
            + BUSINESS_FACTS[
                "full_program_duration"
            ]
            + " formatındadır."
        )

    if topic == "age_range":

        return (
            "Junior Coaching-in əsas qrup proqramı "
            f"{BUSINESS_FACTS['age_min']}–"
            f"{BUSINESS_FACTS['age_max']} yaş üçündür."
        )

    if topic == "language":

        return (
            BUSINESS_FACTS[
                "language"
            ]
        )

    if topic == "format":

        return (
            BUSINESS_FACTS[
                "format"
            ]
        )

    if topic == "payment_model":

        return (
            "Qrup proqramında ödəniş modul üzrə edilir. "
            "Dəqiq modul məbləğləri təsdiqlənmiş source-da "
            "olmadığı üçün rəqəm demirəm."
        )

    if topic == "price":

        child = (
            get_active_child(lead)
            if lead is not None
            else None
        )

        age = (
            child.get("age")
            if child
            else None
        )

        if (
            age is not None
            and age >= 19
        ):

            return (
                "Fərdi coaching görüşü "
                f"{BUSINESS_FACTS['individual_coaching_duration']} "
                "davam edir və bir görüş "
                f"{BUSINESS_FACTS['individual_coaching_price']} AZN-dir."
            )

        return (
            "Qrup proqramının dəqiq modul məbləğləri "
            "hazırda təsdiqlənmiş source-da olmadığı üçün "
            "rəqəm uydurmaq istəmirəm."
        )

    return None


# ============================================================
# 27. ELIGIBILITY
# ============================================================

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
            "Junior Coaching-in əsas qrup proqramı "
            "12–18 yaş üçündür."
        )

    if 12 <= age <= 18:

        return (
            f"{age} yaş Junior Coaching-in "
            "əsas yaş aralığına uyğundur."
        )

    if age == 11:

        return (
            "Hazırda əsas qrup 12–18 yaş üçündür. "
            "12 yaşa çox yaxın olduğu üçün uyğunluğu "
            "mütəxəssislə ayrıca dəqiqləşdirmək olar."
        )

    if age >= 19:

        return (
            "Qrup proqramı 12–18 yaş üçündür. "
            "19 yaş və yuxarı üçün fərdi coaching "
            "daha uyğun istiqamətdir."
        )

    return (
        "Bu yaş əsas Junior Coaching qrupunun "
        "standart yaş aralığından kənardır. "
        "Uyğunluğu mütəxəssislə ayrıca dəqiqləşdirmək lazımdır."
    )


# ============================================================
# 28. MULTI-QUESTION HARD FACT COVERAGE
# ============================================================

def build_hard_fact_answers_from_analysis(
    analysis,
    lead
):

    answers = []

    handled_topics = set()

    topic_map = {

        "location":
            "location",

        "schedule":
            "schedule",

        "group_session_duration":
            "group_session_duration",

        "parent_call_duration":
            "parent_call_duration",

        "child_intro_duration":
            "child_intro_duration",

        "program_duration":
            "program_duration",

        "age_range":
            "age_range",

        "price":
            "price",

        "payment_model":
            "payment_model",

        "language":
            "language",

        "format":
            "format"
    }

    obligations = (
        analysis.get(
            "response_obligations",
            []
        )
        or []
    )

    questions = (
        analysis.get(
            "questions",
            []
        )
        or []
    )

    topics = []

    for obligation in obligations:

        if (
            obligation.get(
                "type"
            )
            == "answer_question"
        ):

            topics.append(
                obligation.get(
                    "topic"
                )
            )

    for question in questions:

        topics.append(
            question.get(
                "topic"
            )
        )

    for topic in topics:

        mapped = topic_map.get(
            topic
        )

        if not mapped:
            continue

        if mapped in handled_topics:
            continue

        answer = hard_fact_answer(
            mapped,
            lead
        )

        if answer:

            answers.append(
                answer
            )

            handled_topics.add(
                mapped
            )

    return answers


# ============================================================
# 29. CONTEXTUAL ANSWER GENERATOR
# ============================================================

RESPONSE_SYSTEM_PROMPT = """
Sən Junior Coaching üzrə AI Sales Assistant və virtual bələdçisən.

FAQ bot deyilsən.

Sənin əsas işin:
- istifadəçini anlamaq
- situasiyanı kontekstdən qiymətləndirmək
- diaqnoz qoymamaq
- istifadəçinin sual və etirazlarına cavab vermək
- sonra yalnız real funksiyası varsa next-step təklif etmək

HARD FACT QAYDASI:

Əgər sənə APPROVED HARD FACTS verilibsə yalnız onlardan istifadə et.

Yaş aralığı, qiymət, müddət, ünvan,
görüş sayı, görüş günü, ödəniş modeli kimi
faktları özündən yaratma.

"5 aylıq proqramdır"
kimi sərt fakt yaratma.

FOUNDATION + LEADERSHIP və ya full-path yalnız
ilkin recommendation kimi ifadə edilə bilər,
business fact kimi yox.

OBJECTION:

Child resistance və spouse skepticism fərqlidir.

Spouse skepticism:
valideynin/yoldaşın proqramın dəyərinə şübhəsidir.

Child resistance:
uşağın özü iştirak etmək istəmir.

Decision dependency:
valideyn qərar üçün həyat yoldaşı ilə danışmalıdır.
Burada lead-i dərhal buraxma.
Əgər blocker aydın deyilsə maksimum 1 qısa sualla
nəyi müzakirə etmək istədiyini dəqiqləşdir.

AMBIGUITY:

Əgər "alınmaya bilər", "bilmirəm mümkün olar ya yox"
kimi qeyri-müəyyənlik varsa səbəbi təxmin etmə.
1 qısa clarification ver.

CONSULTATION:

Valideyn davranış təsvir edirsə keyword axtarma.
Davranışları əlaqələndir.

Amma:
"bu, mütləq özgüvənsizlikdir"
demə.

Belə de:
"bu, konkret mühitdə özünüifadə və özünəinamla
əlaqəli ola bilər"
və ya
"tək bu məlumatla qəti nəticə demək olmaz".

RESPONSE STYLE:

- Default 2–4 qısa cümlə.
- Sadə cavab 20–40 söz.
- İzahlı cavab mümkün qədər 50–70 sözü keçməsin.
- WhatsApp / Instagram üslubu.
- Uzun esse yazma.
- Eyni fikri təkrar etmə.
- Hər cavabın sonunda avtomatik sual vermə.
- Maksimum 1 məqsədli sual.

STATE:

Artıq verilmiş məlumatı yenidən soruşma.
State update istifadəçiyə cavab əvəzi kimi göstərilməməlidir.

Sən yalnız final cavab mətnini yaz.
"""


def generate_contextual_response(
    user_text,
    lead,
    analysis
):

    hits = retrieve_similar(
        user_text,
        top_k=6
    )

    kb_context = [

        {
            "question":
                hit["question"],

            "answer":
                hit["answer"],

            "score":
                round(
                    hit["score"],
                    3
                )
        }

        for hit in hits

        if hit["score"] >= 0.10
    ]

    recommendation = (
        calculate_recommendation(
            lead
        )
    )

    prompt = f"""
USER MESSAGE:
{user_text}

PRIMARY INTENT:
{analysis.get("primary_intent")}

ALL INTENTS:
{json_dumps(analysis.get("all_intents", []))}

RESPONSE OBLIGATIONS:
{json_dumps(analysis.get("response_obligations", []))}

QUESTIONS:
{json_dumps(analysis.get("questions", []))}

OBJECTION:
{json_dumps(analysis.get("objection", {}))}

AMBIGUITY:
{json_dumps(analysis.get("ambiguity", {}))}

NEED ANALYSIS:
{json_dumps(analysis.get("need_analysis", {}))}

STATE:
{json_dumps(sanitize_state_for_llm(lead))}

RECENT CONVERSATION:
{recent_history_text(lead)}

APPROVED HARD FACTS:
{json_dumps(BUSINESS_FACTS)}

KB CANDIDATES:
{json_dumps(kb_context)}

PRELIMINARY RECOMMENDATION:
{recommendation}

Primary intent-i cavablandır.
Bütün response obligations nəzərə alınmalıdır.

Əgər hard fact source-da yoxdursa uydurma.

Əgər ambiguity varsa qərar vermə;
1 clarification ver.

Əgər objection varsa objection-in özünə cavab ver.

Əgər istifadəçi situasiya paylaşırsa
əlaqəli və ehtiyatlı reasoning ver.
"""

    try:

        result = (
            client
            .chat.completions
            .create(

                model=MODEL_NAME,

                temperature=0.20,

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            RESPONSE_SYSTEM_PROMPT
                    },
                    {
                        "role":
                            "user",

                        "content":
                            prompt
                    }
                ]
            )
        )

        return (
            result
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
# 30. OBJECTION HANDLERS
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

    lead[
        "objection_type"
    ] = "child_resistance"

    return (
        "Məcbur etmək tövsiyə olunmur. "
        "Əvvəlcə niyə istəmədiyini anlamaq daha faydalıdır. "
        "Özü razı olsa, proqramı birbaşa eşitməsi üçün "
        "qısa görüntülü tanışlıq təşkil etmək olar."
    )


def handle_spouse_skepticism(
    lead,
    analysis
):

    lead[
        "objection_type"
    ] = "spouse_skepticism"

    objection = (
        analysis.get(
            "objection",
            {}
        )
        or {}
    )

    question = objection.get(
        "best_clarification_question"
    )

    base = (
        "Bu tərəddüd başadüşüləndir. Məqsəd sadəcə "
        "“zamanla öyrənər” deyil, həmin bacarıqları "
        "təhlükəsiz mühitdə praktik şəkildə məşq etdirməkdir."
    )

    if question:

        return (
            base
            + " "
            + question
        )

    return base


def handle_decision_dependency(
    lead,
    analysis
):

    lead[
        "objection_type"
    ] = "decision_dependency"

    # If a real agreed follow-up date already exists,
    # respect it.
    if lead.get(
        "agreed_followup_at"
    ):

        return (
            f"Əlbəttə. {lead['agreed_followup_at']} üçün qeyd edirəm. "
            "O vaxta qədər əlavə follow-up etməyəcəyik."
        )

    objection = (
        analysis.get(
            "objection",
            {}
        )
        or {}
    )

    clarification = (
        objection.get(
            "best_clarification_question"
        )
    )

    if clarification:

        return (
            "Əlbəttə. "
            + clarification
        )

    return (
        "Əlbəttə. Sadəcə dəqiqləşdirim: "
        "yoldaşınızla daha çox proqramın uyğunluğunu "
        "müzakirə etmək istəyirsiniz, yoxsa qiymət və ödəniş tərəfini?"
    )


def handle_ambiguity(
    lead,
    analysis
):

    lead[
        "ambiguity_present"
    ] = True

    lead[
        "clarification_needed"
    ] = True

    ambiguity = (
        analysis.get(
            "ambiguity",
            {}
        )
        or {}
    )

    question = ambiguity.get(
        "best_clarification_question"
    )

    if question:

        return question

    return (
        "Dəqiqləşdirim: çətinlik daha çox vaxt/logistika "
        "baxımındandır, yoxsa iştirakla bağlı başqa tərəddüdünüz var?"
    )


# ============================================================
# 31. READY LEAD
#
# IMPORTANT:
# Ready lead is different from normal lead.
#
# Discovery stops.
# Child intro is mandatory before final acceptance.
# ============================================================

def handle_ready_to_proceed(
    lead
):

    lead[
        "ready_to_proceed"
    ] = True

    lead[
        "sales_stage"
    ] = "READY_TO_PROCEED"

    lead[
        "child_intro_required"
    ] = True

    if (
        lead.get(
            "child_intro_status"
        )
        == "NOT_STARTED"
    ):

        lead[
            "child_intro_status"
        ] = "PENDING"

    lead[
        "next_action"
    ] = "BOOK_CHILD_INTRO"

    if lead.get(
        "child_intro_time"
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

        return (
            "Əla. Növbəti mərhələ övladınızla "
            "təxminən 5 dəqiqəlik görüntülü tanışlıqdır. "
            f"{lead['child_intro_time']} üçün qeyd etdim ✅"
        )

    return (
        "Əla. Discovery-ni burada dayandıraq. "
        "Növbəti mərhələ övladınızla qısa görüntülü tanışlıqdır. "
        "Sizə hansı gün və saat uyğundur?"
    )


# ============================================================
# 32. NORMAL SALES NEXT STEP POLICY
#
# THIS IS THE MAIN V10.2 CHANGE
# ============================================================

def decide_next_step(
    lead,
    analysis
):

    # --------------------------------------------------------
    # 1. HUMAN OWNS LEAD
    # --------------------------------------------------------

    if human_owns_lead(
        lead
    ):

        return None

    # --------------------------------------------------------
    # 2. SAFETY / HANDOFF
    # --------------------------------------------------------

    if lead.get(
        "status"
    ) == "ESCALATED":

        return None

    # --------------------------------------------------------
    # 3. USER HAS A REAL AGREED FOLLOW-UP
    # --------------------------------------------------------

    if lead.get(
        "agreed_followup_at"
    ):

        return None

    # --------------------------------------------------------
    # 4. AMBIGUITY MUST BE RESOLVED FIRST
    # --------------------------------------------------------

    ambiguity = (
        analysis.get(
            "ambiguity",
            {}
        )
        or {}
    )

    if (
        ambiguity.get(
            "present"
        )
        and
        ambiguity.get(
            "clarification_needed"
        )
    ):

        return "CLARIFY_AMBIGUITY"

    # --------------------------------------------------------
    # 5. OBJECTION MUST BE RESOLVED FIRST
    # --------------------------------------------------------

    objection = (
        analysis.get(
            "objection",
            {}
        )
        or {}
    )

    if objection.get(
        "present"
    ):

        objection_type = (
            objection.get(
                "type"
            )
        )

        if objection_type in {
            "spouse_skepticism",
            "decision_dependency",
            "price",
            "schedule",
            "logistics",
            "trust",
            "value"
        }:

            return None

    # --------------------------------------------------------
    # 6. EXPLICIT READY LEAD
    # --------------------------------------------------------

    if lead.get(
        "ready_to_proceed"
    ):

        if (
            lead.get(
                "child_intro_status"
            )
            in {
                "NOT_STARTED",
                "PENDING"
            }
        ):

            return "BOOK_CHILD_INTRO"

        return None

    child = get_active_child(
        lead
    )

    # --------------------------------------------------------
    # 7. AGE
    # --------------------------------------------------------

    if child.get(
        "age"
    ) is None:

        return "ASK_CHILD_AGE"

    age_rule = evaluate_age_rule(
        child
    )

    if age_rule in {
        "SPECIALIST_REVIEW",
        "NOT_STANDARD_GROUP"
    }:

        return "SPECIALIST_REVIEW"

    if age_rule == "INDIVIDUAL":

        return "INDIVIDUAL_PATH"

    # --------------------------------------------------------
    # 8. NEED
    # --------------------------------------------------------

    if not child.get(
        "need"
    ):

        return "ASK_NEED"

    # --------------------------------------------------------
    # 9. ONE USEFUL CONSULTATIVE QUESTION
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 10. CONTACT
    # --------------------------------------------------------

    if (
        not lead.get(
            "parent_name"
        )
        and
        not lead.get(
            "phone"
        )
    ):

        return "ASK_NAME_PHONE"

    if not lead.get(
        "phone"
    ):

        return "ASK_PHONE"

    # --------------------------------------------------------
    # 11. NORMAL LEAD -> PARENT CALL
    #
    # THIS USED TO INCORRECTLY BOOK CHILD VIDEO.
    # --------------------------------------------------------

    if (
        lead.get(
            "parent_call_status"
        )
        == "NOT_STARTED"
    ):

        return "ASK_PARENT_CALL_TIME"

    if (
        lead.get(
            "parent_call_status"
        )
        == "PENDING"
    ):

        return "ASK_PARENT_CALL_TIME"

    # --------------------------------------------------------
    # 12. PARENT CALL BOOKED
    # --------------------------------------------------------

    if (
        lead.get(
            "parent_call_status"
        )
        == "BOOKED"
    ):

        return None

    # --------------------------------------------------------
    # 13. AFTER PARENT CALL
    #
    # Child intro is NOT automatic.
    # It must be required/requested/ready.
    # --------------------------------------------------------

    if (
        lead.get(
            "parent_call_status"
        )
        == "COMPLETED"
    ):

        if (
            lead.get(
                "child_intro_required"
            )
            or
            analysis.get(
                "child_intro_requested"
            )
        ):

            return "BOOK_CHILD_INTRO"

        return None

    return None


# ============================================================
# 33. RENDER NEXT STEP
# ============================================================

def render_next_step(
    lead,
    analysis
):

    action = decide_next_step(
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

    if action == "CLARIFY_AMBIGUITY":

        return handle_ambiguity(
            lead,
            analysis
        )

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
            "dəyişməsini və ya inkişaf etməsini istəyirsiniz?"
        )

    if action == "CLARIFY_NEED":

        need_analysis = (
            analysis.get(
                "need_analysis",
                {}
            )
            or {}
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

    if action == "ASK_PARENT_CALL_TIME":

        lead[
            "parent_call_status"
        ] = "PENDING"

        lead[
            "sales_stage"
        ] = "PARENT_CALL_PENDING"

        lead[
            "_last_question_topic"
        ] = "parent_call_time"

        return (
            "Valideynlə 5–7 dəqiqəlik ilkin telefon danışığı üçün "
            "sizə hansı gün və saat aralığı daha uyğundur?"
        )

    if action == "BOOK_CHILD_INTRO":

        lead[
            "child_intro_required"
        ] = True

        lead[
            "child_intro_status"
        ] = "PENDING"

        lead[
            "_last_question_topic"
        ] = "child_intro_time"

        return (
            "Övladınızla təxminən 5 dəqiqəlik görüntülü tanışlıq üçün "
            "sizə hansı gün və saat aralığı uyğundur?"
        )

    if action == "SPECIALIST_REVIEW":

        return (
            "Bu yaş standart qrup aralığından kənar olduğu üçün "
            "uyğunluğu mütəxəssislə ayrıca dəqiqləşdirmək daha doğru olar."
        )

    if action == "INDIVIDUAL_PATH":

        return (
            "Bu yaş üçün əsas qrup proqramından çox "
            "fərdi coaching istiqaməti nəzərdən keçirilə bilər."
        )

    return None


# ============================================================
# 34. SHOULD WE APPEND FLOW?
#
# We do NOT append flow blindly.
# ============================================================

def should_append_next_step(
    lead,
    analysis
):

    primary = analysis.get(
        "primary_intent"
    )

    # --------------------------------------------------------
    # Don't append after these
    # --------------------------------------------------------

    if primary in {

        "state_recall",
        "followup_commitment",
        "human_request",
        "complaint",
        "clinical_risk",
        "partnership",
        "special_payment",
        "spouse_skepticism",
        "decision_dependency",
        "ambiguous_objection",
        "child_resistance"

    }:

        return False

    ambiguity = (
        analysis.get(
            "ambiguity",
            {}
        )
        or {}
    )

    if ambiguity.get(
        "present"
    ):

        return False

    objection = (
        analysis.get(
            "objection",
            {}
        )
        or {}
    )

    if objection.get(
        "present"
    ):

        return False

    # --------------------------------------------------------
    # If the user asked FAQ only and hasn't engaged as a lead,
    # do not force qualification.
    # --------------------------------------------------------

    child = get_active_child(
        lead
    )

    has_lead_context = any([

        child.get(
            "age"
        ) is not None,

        bool(
            child.get("need")
        ),

        bool(
            lead.get("phone")
        ),

        bool(
            lead.get(
                "parent_name"
            )
        )
    ])

    if (
        primary in {
            "location",
            "schedule",
            "duration",
            "language",
            "program_info",
            "price"
        }
        and
        not has_lead_context
    ):

        return False

    return True


def append_next_step_if_needed(
    response,
    lead,
    analysis
):

    if not should_append_next_step(
        lead,
        analysis
    ):

        return response

    next_text = render_next_step(
        lead,
        analysis
    )

    if not next_text:

        return response

    if (
        normalize_text(
            next_text
        )
        in
        normalize_text(
            response
        )
    ):

        return response

    return (
        response.rstrip()
        +
        "\n\n"
        +
        next_text
    )


# ============================================================
# 35. ANSWER MULTI-PART MESSAGE
# ============================================================

def answer_multi_part_message(
    user_text,
    lead,
    analysis
):

    hard_fact_answers = (
        build_hard_fact_answers_from_analysis(
            analysis,
            lead
        )
    )

    primary = analysis.get(
        "primary_intent"
    )

    # If ALL obligations were approved hard facts,
    # deterministic answers are enough.

    obligations = (
        analysis.get(
            "response_obligations",
            []
        )
        or []
    )

    question_obligations = [

        item

        for item in obligations

        if item.get(
            "type"
        ) == "answer_question"
    ]

    if (
        hard_fact_answers
        and
        len(hard_fact_answers)
        >= len(question_obligations)
        and
        primary not in {
            "consultation",
            "spouse_skepticism",
            "decision_dependency",
            "child_resistance",
            "ambiguous_objection"
        }
    ):

        return "\n\n".join(
            hard_fact_answers
        )

    # Otherwise let LLM combine contextual + factual obligations,
    # while passing approved facts.

    return generate_contextual_response(
        user_text,
        lead,
        analysis
    )


# ============================================================
# 36. MAIN AGENT
# ============================================================

def lead_agent_reply(
    user_text,
    lead,
    faq_min_score=0.25,
    history=None,
    conversation_history=None
):

    # --------------------------------------------------------
    # app.py backward compatibility
    # --------------------------------------------------------

    if conversation_history is None:

        conversation_history = history

    user_text = compact_spaces(
        user_text
    )

    if not user_text:

        return (
            "Mesajınızı yaza bilərsiniz."
        )

    ensure_children(
        lead
    )

    # --------------------------------------------------------
    # Import app conversation history only once
    # --------------------------------------------------------

    if (
        conversation_history
        and
        not lead.get(
            "_conversation_history"
        )
    ):

        normalized_history = []

        for item in (
            conversation_history[-12:]
        ):

            if not isinstance(
                item,
                dict
            ):
                continue

            role = item.get(
                "role"
            )

            content = item.get(
                "content"
            )

            if (
                role in {
                    "user",
                    "assistant"
                }
                and content
            ):

                normalized_history.append({

                    "role":
                        role,

                    "content":
                        content
                })

        lead[
            "_conversation_history"
        ] = normalized_history

    # --------------------------------------------------------
    # HARD HUMAN OWNERSHIP STOP
    # --------------------------------------------------------

    if human_owns_lead(
        lead
    ):

        response = (
            "Müraciətiniz artıq əməkdaşımıza yönləndirilib. "
            "Paralel olaraq əlavə qualification və satış mesajı göndərməyəcəyəm."
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

    # --------------------------------------------------------
    # UNDERSTAND
    # --------------------------------------------------------

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

            "objection":
                (
                    analysis.get(
                        "objection",
                        {}
                    )
                    or {}
                ).get(
                    "type"
                ),

            "ambiguity":
                (
                    analysis.get(
                        "ambiguity",
                        {}
                    )
                    or {}
                ).get(
                    "present"
                ),

            "confidence":
                analysis.get(
                    "confidence"
                )
        }
    )

    # --------------------------------------------------------
    # EXTRACT + UPDATE STATE
    # --------------------------------------------------------

    apply_analysis_to_state(
        lead,
        analysis
    )

    calculate_recommendation(
        lead
    )

    primary = analysis.get(
        "primary_intent",
        "other"
    )

    # ========================================================
    # SAFETY / BUSINESS RULES
    # ========================================================

    if (
        analysis.get(
            "clinical_or_safety_risk"
        )
        or
        primary == "clinical_risk"
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

    # --------------------------------------------------------
    # COMPLAINT
    # --------------------------------------------------------

    elif (
        analysis.get(
            "complaint"
        )
        or
        primary == "complaint"
    ):

        mark_handoff(
            lead
        )

        response = (
            "Narazılığınızı başa düşürəm. "
            "Məsələnin düzgün araşdırılması üçün "
            "müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # --------------------------------------------------------
    # HUMAN REQUEST
    # --------------------------------------------------------

    elif (
        analysis.get(
            "human_requested"
        )
        or
        primary == "human_request"
    ):

        mark_handoff(
            lead
        )

        if lead.get(
            "phone"
        ):

            response = (
                "Əlbəttə. Müraciətinizi əməkdaşımıza yönləndirirəm. "
                "Əlaqə nömrəniz artıq qeyd olunub."
            )

        else:

            response = (
                "Əlbəttə. Müraciətinizi əməkdaşımıza yönləndirə bilərəm. "
                "Əlaqə nömrənizi yaza bilərsiniz?"
            )

    # --------------------------------------------------------
    # PARTNERSHIP
    # --------------------------------------------------------

    elif (
        analysis.get(
            "partnership"
        )
        or
        primary == "partnership"
    ):

        mark_handoff(
            lead
        )

        response = (
            "Əməkdaşlıq təkliflərini aidiyyəti komanda dəyərləndirir. "
            "Müraciətinizi həmin komandaya yönləndirmək daha doğru olar."
        )

    # --------------------------------------------------------
    # SPECIAL PAYMENT
    # --------------------------------------------------------

    elif (
        analysis.get(
            "special_payment_request"
        )
        or
        primary == "special_payment"
    ):

        response = (
            "Standart qaydada ödəniş modul üzrə edilir. "
            "Əgər başlamağa əsas maneə ödəniş formasıdırsa, "
            "bunu rəhbərliklə ayrıca dəqiqləşdirmək olar."
        )

    # ========================================================
    # STATE RECALL
    # ========================================================

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

        recall = (
            analysis.get(
                "state_recall",
                {}
            )
            or {}
        )

        response = answer_state_recall(
            lead,
            recall.get(
                "fields",
                []
            )
        )

    # ========================================================
    # AMBIGUITY
    # ========================================================

    elif (
        primary == "ambiguous_objection"
        or
        (
            analysis.get(
                "ambiguity",
                {}
            )
            or {}
        ).get(
            "present"
        )
    ):

        response = handle_ambiguity(
            lead,
            analysis
        )

    # ========================================================
    # CHILD RESISTANCE
    # ========================================================

    elif (
        primary == "child_resistance"
        or
        (
            analysis.get(
                "objection",
                {}
            )
            or {}
        ).get(
            "type"
        )
        == "child_resistance"
    ):

        response = handle_child_resistance(
            lead
        )

    # ========================================================
    # SPOUSE SKEPTICISM
    # ========================================================

    elif (
        primary == "spouse_skepticism"
        or
        (
            analysis.get(
                "objection",
                {}
            )
            or {}
        ).get(
            "type"
        )
        == "spouse_skepticism"
    ):

        response = (
            handle_spouse_skepticism(
                lead,
                analysis
            )
        )

    # ========================================================
    # DECISION DEPENDENCY
    # ========================================================

    elif (
        primary == "decision_dependency"
        or
        (
            analysis.get(
                "objection",
                {}
            )
            or {}
        ).get(
            "type"
        )
        == "decision_dependency"
    ):

        response = (
            handle_decision_dependency(
                lead,
                analysis
            )
        )

    # ========================================================
    # EXPLICIT AGREED FOLLOWUP
    # ========================================================

    elif (
        primary == "followup_commitment"
        and
        lead.get(
            "agreed_followup_at"
        )
    ):

        lead[
            "next_action"
        ] = "AGREED_FOLLOWUP"

        response = (
            f"Əlbəttə. {lead['agreed_followup_at']} üçün qeyd edirəm. "
            "O vaxta qədər əlavə follow-up etməyəcəyik."
        )

    # ========================================================
    # READY TO PROCEED
    # ========================================================

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

    # ========================================================
    # ELIGIBILITY
    # ========================================================

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

    # ========================================================
    # SIMPLE HARD FACT
    # ========================================================

    elif primary in {
        "location",
        "schedule",
        "price",
        "language",
        "duration"
    }:

        # More than one question?
        if (
            len(
                analysis.get(
                    "questions",
                    []
                )
            )
            > 1
        ):

            response = (
                answer_multi_part_message(
                    user_text,
                    lead,
                    analysis
                )
            )

        else:

            topic = primary

            if primary == "duration":

                # Need exact duration subtype.
                questions = (
                    analysis.get(
                        "questions",
                        []
                    )
                    or []
                )

                if questions:

                    topic = questions[0].get(
                        "topic",
                        "group_session_duration"
                    )

                else:

                    topic = (
                        "group_session_duration"
                    )

            response = hard_fact_answer(
                topic,
                lead
            )

            if not response:

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

    # ========================================================
    # GREETING
    # ========================================================

    elif primary == "greeting":

        if lead.get(
            "_conversation_history"
        ):

            response = "Salam 😊"

        else:

            response = (
                "Salam 😊 Sizə necə kömək edə bilərəm?"
            )

    # ========================================================
    # MULTI-PART QUESTIONS
    # ========================================================

    elif (
        len(
            analysis.get(
                "response_obligations",
                []
            )
        )
        > 1
    ):

        response = (
            answer_multi_part_message(
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

    # ========================================================
    # CONSULTATION / PROGRAM INFO / OTHER QUESTION
    # ========================================================

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

    # ========================================================
    # USER PROVIDED INFORMATION
    # ========================================================

    else:

        next_text = render_next_step(
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
                    analysis
                )
            )

    # ========================================================
    # BOOK NORMAL PARENT CALL
    #
    # If parent has given a time while parent-call is pending.
    # ========================================================

    if (
        not lead.get(
            "ready_to_proceed"
        )
        and
        lead.get(
            "parent_call_time"
        )
        and
        lead.get(
            "phone"
        )
        and
        lead.get(
            "parent_call_status"
        )
        in {
            "NOT_STARTED",
            "PENDING"
        }
    ):

        lead[
            "parent_call_status"
        ] = "BOOKED"

        lead[
            "sales_stage"
        ] = "PARENT_CALL_BOOKED"

        lead[
            "next_action"
        ] = "PARENT_CALL"

        # Compatibility with current Streamlit completion logic
        lead[
            "status"
        ] = "CALL_REQUESTED"

    # ========================================================
    # BOOK READY LEAD CHILD INTRO
    # ========================================================

    if (
        lead.get(
            "ready_to_proceed"
        )
        and
        lead.get(
            "child_intro_time"
        )
        and
        lead.get(
            "child_intro_status"
        )
        in {
            "NOT_STARTED",
            "PENDING"
        }
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

    # ========================================================
    # MEMORY
    # ========================================================

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

    # Current app.py expects only string
    return response


# ============================================================
# 37. OLD APP / CLI COMPATIBILITY
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

    if (
        lead.get(
            "parent_call_status"
        )
        in {
            "NOT_STARTED",
            "PENDING"
        }
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
            "dəyişməsini və ya inkişaf etməsini istəyirsiniz?"
        ),

    "phone":
        (
            "Sizinlə əlaqə saxlaya bilməyimiz üçün "
            "telefon nömrənizi yaza bilərsiniz?"
        ),

    "preferred_call_time":
        (
            "Valideynlə 5–7 dəqiqəlik ilkin telefon danışığı üçün "
            "sizə hansı gün və saat aralığı uyğundur?"
        )
}


def answer_faq_question(
    question,
    min_score=0.25
):

    hit = get_best_faq_hit(
        question,
        min_score
    )

    if not hit:

        return (
            "Bu sualla bağlı təsdiqlənmiş məlumat bazasında "
            "dəqiq cavab tapa bilmədim."
        )

    return hit[
        "answer"
    ]


# ============================================================
# 38. CONVERSATION LOG
#
# Backward compatible with old app.py
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

    # Old:
    # save_conversation_log(session, user, bot, lead)

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

        faq_score = (
            lead.get(
                "_last_faq_score"
            )
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
            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?, ?
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
# 39. ADMIN HELPERS
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