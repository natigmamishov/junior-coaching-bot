"""
Junior Coaching — Conversation Engine V6.1

Əsas prinsiplər:
1. İstifadəçinin bütün mesajı analiz olunur.
2. Bir mesajdan bir neçə məlumat eyni anda çıxarıla bilər.
3. Bir mesajda bir neçə sual varsa ayrıca emal edilir.
4. Əvvəl verilmiş məlumat təkrar soruşulmur.
5. İstifadəçi məlumatı düzəldirsə state yenilənir.
6. İstifadəçi arada sual verirsə əvvəl suala cavab verilir.
7. Hər turn-də maksimum 1 flow sualı verilir.
8. FAQ-only istifadəçiyə yumşaq conversation bridge göstərilir.
9. Salamlaşma təkrar-təkrar edilmir.
10. SQLite və mövcud Streamlit app ilə uyğunluq saxlanılır.
"""

import os
import re
import json
import sqlite3

from datetime import datetime
from typing import Optional, List, Dict, Any
from zoneinfo import ZoneInfo

import httpx
import numpy as np

from dotenv import load_dotenv
from openai import OpenAI

from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 0. PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Junior_Coaching_sesli_AI_FAQ.txt"
)

DB_PATH = os.path.join(
    BASE_DIR,
    "junior_coaching.db"
)


# =========================================================
# 1. OPENAI
# =========================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env"
    )
)

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client: Optional[OpenAI] = None

if OPENAI_API_KEY:

    http_client = httpx.Client(
        verify=False,
        timeout=60,
    )

    client = OpenAI(
        api_key=OPENAI_API_KEY,
        http_client=http_client,
    )


# =========================================================
# 2. NORMALIZATION
# =========================================================

AZ_TRANSLATION = str.maketrans({
    "ə": "e",
    "Ə": "e",
    "ı": "i",
    "İ": "i",
    "ö": "o",
    "Ö": "o",
    "ü": "u",
    "Ü": "u",
    "ş": "s",
    "Ş": "s",
    "ç": "c",
    "Ç": "c",
    "ğ": "g",
    "Ğ": "g",
})


def normalize_text(text: str) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def normalize_for_search(text: str) -> str:

    text = normalize_text(text)

    text = text.translate(
        AZ_TRANSLATION
    )

    text = re.sub(
        r"[^\w\s?]",
        " ",
        text,
    )

    return re.sub(
        r"\s+",
        " ",
        text,
    ).strip()


# =========================================================
# 3. BASIC HELPERS
# =========================================================

def normalize_phone(text: str) -> Optional[str]:

    digits = re.sub(
        r"\D",
        "",
        str(text)
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


def extract_all_ages(text: str) -> List[int]:

    values = re.findall(
        r"\b\d{1,2}\b",
        str(text)
    )

    result = []

    for value in values:

        number = int(value)

        if (
            1 <= number <= 99
            and number not in result
        ):
            result.append(number)

    return result


def is_greeting(text: str) -> bool:

    value = normalize_for_search(text)

    return value in {
        "salam",
        "slm",
        "salamlar",
        "hello",
        "hi",
        "hey",
        "salam necesiz",
        "salam necesiniz",
        "salam aleykum",
        "aleykum salam",
    }


# =========================================================
# 4. NAME CLEANING
# =========================================================

NON_NAME_TOKENS = {
    "men",
    "mene",
    "mən",
    "mənə",
    "anasi",
    "anasiyam",
    "atasiyam",
    "atasi",
    "valideynem",
    "valideyniyem",
    "oglum",
    "qizim",
    "usaq",
    "usaqdir",
    "ovladim",
    "dedim",
    "deyirem",
    "yuxarida",
    "salam",
    "sagol",
    "sagolun",
    "tesekkur",
    "tesekkurler",
    "adim",
    "adım",
    "adi",
    "adı",
}


def remove_honorific(text: str) -> str:

    words = text.strip().split()

    blocked = {
        "bey",
        "bəy",
        "xanim",
        "xanım",
        "muellim",
        "müəllim",
    }

    result = []

    for word in words:

        normalized = normalize_for_search(word)

        if normalized not in {
            normalize_for_search(x)
            for x in blocked
        }:
            result.append(word)

    return " ".join(result)


def remove_name_suffix(word: str) -> str:

    if not word:
        return word

    normalized = normalize_for_search(word)

    suffixes = [
        "dir",
        "dur",
    ]

    for suffix in suffixes:

        if normalized.endswith(suffix):

            base_length = (
                len(word)
                - len(suffix)
            )

            if base_length >= 4:

                return word[:base_length]

    return word


def clean_name(value: str) -> Optional[str]:

    if not value:
        return None

    value = remove_honorific(value)

    value = re.sub(
        r"[.,!?+():;]",
        "",
        value,
    ).strip()

    if not value:
        return None

    result_words = []

    for word in value.split():

        normalized_word = normalize_for_search(word)

        if normalized_word in {
            normalize_for_search(x)
            for x in NON_NAME_TOKENS
        }:
            continue

        cleaned = remove_name_suffix(word)

        if cleaned:
            result_words.append(cleaned)

    if not result_words:
        return None

    result_words = result_words[:2]

    result = " ".join(result_words)

    if not re.fullmatch(
        r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+",
        result,
    ):
        return None

    return result.title()


def deterministic_parent_name_extract(
    text: str
) -> Optional[str]:

    patterns = [
        r"(?i)\bmənim\s+adım\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bmenim\s+adim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\badım\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\badim\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)",
        r"(?i)\bmən\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)[əa]m\b",
        r"(?i)\bmen\s+([A-Za-zƏəÖöÜüĞğÇçŞşİı\-]+)[ae]m\b",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
        )

        if match:

            return clean_name(
                match.group(1)
            )

    return None


# =========================================================
# 5. AZERBAIJANI NAME SUFFIX HELPERS
# =========================================================

def get_last_vowel(word: str) -> Optional[str]:

    vowels = "aıoueəiöü"

    for char in reversed(
        word.lower()
    ):

        if char in vowels:
            return char

    return None


def get_genitive_suffix(word: str) -> str:

    vowel = get_last_vowel(word)

    if vowel in ["a", "ı"]:
        return "ın"

    if vowel in ["e", "ə", "i"]:
        return "in"

    if vowel in ["o", "u"]:
        return "un"

    if vowel in ["ö", "ü"]:
        return "ün"

    return "ın"


def child_genitive(name: str) -> str:

    if not name:
        return "övladınızın"

    suffix = get_genitive_suffix(name)

    if name[-1].lower() in "aıoueəiöü":

        return (
            name
            + "n"
            + suffix
        )

    return (
        name
        + suffix
    )


# =========================================================
# 6. LEAD STRUCTURE
# =========================================================

def create_empty_child() -> dict:

    return {
        "name": None,
        "age": None,
        "main_concern": None,
        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,
    }


def create_empty_lead(
    source: str = "CLI"
) -> dict:

    return {
        "parent_name": None,
        "parent_title": None,

        "child_name": None,
        "child_age": None,
        "main_concern": None,

        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,

        "children": [
            create_empty_child()
        ],

        "active_child_index": 0,
        "multiple_children": False,

        "phone": None,
        "preferred_call_time": None,

        "source": source,
        "status": "NEW",

        "_last_intent": None,
        "_last_confidence": None,
        "_last_faq_score": None,

        "_greeted": False,
        "_flow_started": False,
        "_last_answer_topic": None,
        "_repeat_topic_count": 0,
    }


def ensure_lead_structure(
    lead: dict
):

    if "children" not in lead:

        lead["children"] = [
            create_empty_child()
        ]

    if not lead["children"]:

        lead["children"] = [
            create_empty_child()
        ]

    lead.setdefault(
        "active_child_index",
        0
    )

    lead.setdefault(
        "multiple_children",
        False
    )

    lead.setdefault(
        "_greeted",
        False
    )

    lead.setdefault(
        "_flow_started",
        False
    )

    lead.setdefault(
        "_last_answer_topic",
        None
    )

    lead.setdefault(
        "_repeat_topic_count",
        0
    )


def get_active_child(
    lead: dict
) -> dict:

    ensure_lead_structure(lead)

    index = lead.get(
        "active_child_index",
        0
    )

    if index >= len(
        lead["children"]
    ):

        index = 0

        lead[
            "active_child_index"
        ] = 0

    return lead[
        "children"
    ][index]


def sync_flat_fields(
    lead: dict
):

    ensure_lead_structure(lead)

    first = lead["children"][0]

    lead["child_name"] = first.get(
        "name"
    )

    lead["child_age"] = first.get(
        "age"
    )

    lead["main_concern"] = first.get(
        "main_concern"
    )

    lead[
        "needs_concern_followup"
    ] = first.get(
        "needs_concern_followup",
        False
    )

    lead[
        "concern_duration"
    ] = first.get(
        "concern_duration"
    )

    lead[
        "concern_onset"
    ] = first.get(
        "concern_onset"
    )


# =========================================================
# 7. TITLE
# =========================================================

def infer_parent_title_with_llm(
    parent_name: str
) -> str:

    if (
        not parent_name
        or client is None
    ):
        return ""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": """
Azərbaycan adına əsasən müraciət formasını müəyyən et.

Yalnız:
xanım
bəy
neutral

Əmin deyilsənsə neutral seç.
"""
                },
                {
                    "role": "user",
                    "content": parent_name,
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "title_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "title": {
                                "type": "string",
                                "enum": [
                                    "xanım",
                                    "bəy",
                                    "neutral",
                                ],
                            }
                        },
                        "required": [
                            "title"
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        result = json.loads(
            response.choices[0].message.content
        )

        title = result.get("title")

        if title in [
            "xanım",
            "bəy",
        ]:
            return title

    except Exception as exc:

        print(
            "TITLE ERROR:",
            exc
        )

    return ""


def get_parent_display_name(
    lead: dict
) -> str:

    name = lead.get(
        "parent_name"
    )

    title = lead.get(
        "parent_title"
    )

    if not name:
        return ""

    if title:

        return (
            f"{name} {title}"
        )

    return name


# =========================================================
# 8. FAQ INDEX
# =========================================================

def build_faq_index():

    if not os.path.exists(
        DATASET_PATH
    ):

        raise FileNotFoundError(
            DATASET_PATH
        )

    with open(
        DATASET_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        raw = file.read()

    pattern = re.compile(
        r"(?:\d+\.\s*)?Sual:\s*(.*?)\s*"
        r"(?:Agent|Selnaz|Cavab):\s*(.*?)"
        r"(?=\n\s*(?:\d+\.\s*)?Sual:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    pairs = []

    for question, answer in pattern.findall(
        raw
    ):

        question = re.sub(
            r"\s+",
            " ",
            question,
        ).strip()

        answer = re.sub(
            r"\s+",
            " ",
            answer,
        ).strip()

        if question and answer:

            pairs.append(
                (
                    question,
                    answer,
                )
            )

    if not pairs:

        raise ValueError(
            "FAQ faylında sual-cavab tapılmadı."
        )

    questions = [
        q
        for q, _ in pairs
    ]

    answers = [
        a
        for _, a in pairs
    ]

    normalized_questions = [
        normalize_for_search(q)
        for q in questions
    ]

    vectorizer = FeatureUnion([

        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
                max_features=60000,
            )
        ),

        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
                max_features=80000,
            )
        ),

    ])

    matrix = vectorizer.fit_transform(
        normalized_questions
    )

    return (
        questions,
        answers,
        vectorizer,
        matrix,
    )


(
    FAQ_QUESTIONS,
    FAQ_ANSWERS,
    FAQ_VECTORIZER,
    FAQ_MATRIX,
) = build_faq_index()


def retrieve_faq_candidates(
    query: str,
    k: int = 5,
) -> List[Dict[str, Any]]:

    normalized_query = normalize_for_search(
        query
    )

    vector = FAQ_VECTORIZER.transform(
        [normalized_query]
    )

    scores = cosine_similarity(
        vector,
        FAQ_MATRIX,
    ).ravel()

    top_indices = np.argsort(
        -scores
    )[:k]

    result = []

    for index in top_indices:

        result.append({
            "index": int(index),
            "question": FAQ_QUESTIONS[index],
            "answer": FAQ_ANSWERS[index],
            "score": float(scores[index]),
        })

    return result


# =========================================================
# 9. SEMANTIC FAQ SELECTION
# =========================================================

def select_best_faq_with_llm(
    question: str,
    candidates: List[Dict[str, Any]],
):

    if not candidates:
        return None

    if client is None:

        best = candidates[0]

        if best["score"] >= 0.18:
            return best

        return None

    candidate_text = "\n\n".join([
        (
            f"ID={i}\n"
            f"Sual: {item['question']}\n"
            f"Cavab: {item['answer']}"
        )
        for i, item in enumerate(
            candidates
        )
    ])

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": """
İstifadəçinin sualına semantik olaraq ən uyğun FAQ-nı seç.

ÇOX VACİB:
- Açar sözə deyil, mənaya bax.
- "telefon zəngi neçə dəqiqədir?" ilə "proqram neçə aydır?" fərqlidir.
- "görüş harada keçirilir?" ilə "görüş nə vaxt keçirilir?" fərqlidir.
- "buraxılan görüş əvəzlənir?" ilə "bir dəfəlik görüşə gəlmək olar?" fərqlidir.
- Heç biri dəqiq uyğun deyilsə -1 qaytar.
"""
                },
                {
                    "role": "user",
                    "content": (
                        f"İstifadəçi sualı:\n{question}\n\n"
                        f"FAQ namizədləri:\n{candidate_text}"
                    ),
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "faq_selection",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "selected_id": {
                                "type": "integer"
                            },
                            "confidence": {
                                "type": "number"
                            },
                        },
                        "required": [
                            "selected_id",
                            "confidence",
                        ],
                        "additionalProperties": False,
                    },
                },
            },
        )

        result = json.loads(
            response.choices[0].message.content
        )

        selected_id = result.get(
            "selected_id",
            -1
        )

        confidence = result.get(
            "confidence",
            0
        )

        if (
            0 <= selected_id < len(candidates)
            and confidence >= 0.55
        ):

            return candidates[
                selected_id
            ]

    except Exception as exc:

        print(
            "FAQ SELECT ERROR:",
            exc
        )

    best = candidates[0]

    if best["score"] >= 0.25:
        return best

    return None


# =========================================================
# 10. SPECIAL QUESTIONS
# =========================================================

def is_permission_to_ask(
    text: str
) -> bool:

    value = normalize_for_search(
        text
    )

    patterns = [
        "bir sual vere bilerem",
        "bir sual verim",
        "sual vere bilerem",
        "sizden bir sey sorusum",
        "bir sey sorusa bilerem",
    ]

    return any(
        pattern in value
        for pattern in patterns
    )


def is_child_presence_question(
    text: str
) -> bool:

    value = normalize_for_search(
        text
    )

    return (
        any(
            x in value
            for x in [
                "usaq",
                "ovlad",
                "oglum",
                "qizim",
            ]
        )
        and any(
            x in value
            for x in [
                "yanimda",
                "olmalidir",
                "gelmelidir",
                "istirak",
                "zengde",
            ]
        )
    )


def is_contact_here_question(
    text: str
) -> bool:

    value = normalize_for_search(
        text
    )

    return (
        any(
            x in value
            for x in [
                "burdan",
                "buradan",
                "burda",
                "burada",
            ]
        )
        and any(
            x in value
            for x in [
                "elaqe",
                "danismaq",
                "yazismaq",
                "mumkundur",
                "olar",
            ]
        )
    )


def is_bot_question(
    text: str
) -> bool:

    value = normalize_for_search(
        text
    )

    patterns = [
        "siz botsuz",
        "sen botsan",
        "kimle danisiram",
        "siz kimsiniz",
        "kim cavab verir",
    ]

    return any(
        pattern in value
        for pattern in patterns
    )


def is_state_question(
    text: str
) -> Optional[str]:

    value = normalize_for_search(
        text
    )

    if any(
        x in value
        for x in [
            "adimi qeyd etdiniz",
            "adimi yazdiniz",
            "adimi goturdunuz",
            "menim adim ne idi",
        ]
    ):
        return "parent_name"

    if any(
        x in value
        for x in [
            "usaqin adini qeyd etdiniz",
            "ovladimin adini qeyd etdiniz",
            "usaqin adi ne idi",
        ]
    ):
        return "child_name"

    if any(
        x in value
        for x in [
            "yasini qeyd etdiniz",
            "usaqin yasini qeyd etdiniz",
            "yasi yadinizdadir",
        ]
    ):
        return "child_age"

    if any(
        x in value
        for x in [
            "nomremi qeyd etdiniz",
            "telefonumu qeyd etdiniz",
            "nomremi goturdunuz",
        ]
    ):
        return "phone"

    return None


# =========================================================
# 11. WHOLE MESSAGE ANALYSIS
# =========================================================

def analyze_message(
    user_text: str,
    lead: dict,
    history: Optional[List[dict]] = None,
) -> dict:

    ensure_lead_structure(
        lead
    )

    history = history or []

    compact_history = history[-8:]

    state = {
        "parent_name": lead.get(
            "parent_name"
        ),
        "parent_title": lead.get(
            "parent_title"
        ),
        "children": lead.get(
            "children"
        ),
        "phone": lead.get(
            "phone"
        ),
        "preferred_call_time": lead.get(
            "preferred_call_time"
        ),
        "status": lead.get(
            "status"
        ),
    }

    if client is None:

        return {
            "intent": (
                "greeting"
                if is_greeting(user_text)
                else "field_answer"
            ),
            "questions": (
                [user_text]
                if "?" in user_text
                else []
            ),
            "parent_name": "",
            "child_name": "",
            "child_age": 0,
            "main_concern": "",
            "phone": "",
            "preferred_call_time": "",
            "multiple_children": False,
            "children_count": 0,
            "corrections": [],
            "confidence": 0.0,
        }

    system_prompt = """
Sən Junior Coaching üçün conversation analyzer-sən.

Sənin vəzifən:
- cari user mesajını,
- əvvəlki söhbət tarixçəsini,
- artıq toplanmış state-i

birlikdə nəzərə almaqdır.

Bir mesajda bir neçə məlumat varsa HAMISINI çıxar.
Bir mesajda bir neçə sual varsa HAMISINI questions array-də ayrıca yaz.

Məsələn:
"Görüşlər harada keçirilir və telefon zəngi neçə dəqiqədir?"
questions:
[
  "Görüşlər harada keçirilir?",
  "Telefon zəngi neçə dəqiqədir?"
]

Məsələn:
"Mən Nərgizəm. Oğlum Orxanın 15 yaşı var.
Özgüvəni zəifdir. Nömrəm 0501234567.
Sabah 15:00-dan sonra danışa bilərəm."

parent_name=Nərgiz
child_name=Orxan
child_age=15
main_concern=özgüvən
phone=0501234567
preferred_call_time=sabah 15:00-dan sonra

ƏSAS QAYDALAR:

1. "məsuliyyətsizdir", "özgüvəni zəifdir",
"məqsəd və gələcək", "ünsiyyəti zəifdir",
"hamısı", "hər biri" complaint deyil.
Bunlar child main_concern-dir.

2. "Aygün mənəm, uşağın adı Ayxandır"
corrections daxilində:
parent_name=Aygün
child_name=Ayxan

3. "mənə yox, Tunar"
əgər əvvəl child name müzakirə olunurdusa child_name=Tunar correction.

4. User artıq verilmiş məlumatı düzəldirsə correction qaytar.

5. "Adımı qeyd etdiniz?" FAQ deyil.
state_question intent istifadə et.

6. "Bir sual verə bilərəm?"
permission_question intent istifadə et.

7. "anasıyam", "mənə", "atasıyam" ad deyil.

8. İki uşaq varsa multiple_children=true.

9. Uşaq barədə mənfi xüsusiyyət service complaint deyil.
Complaint yalnız Junior Coaching xidmətinə narazılıqdır.

10. User eyni anda sual + məlumat verə bilər.
Həm sualı, həm slotları çıxar.

Intent-lər:
greeting
faq_question
field_answer
program_interest
registration_request
state_question
permission_question
human_agent_request
complaint
safety_risk
meta_question
pause_request
unrelated
"""

    prompt = f"""
CURRENT STATE:
{json.dumps(state, ensure_ascii=False)}

RECENT HISTORY:
{json.dumps(compact_history, ensure_ascii=False)}

CURRENT USER MESSAGE:
{user_text}
"""

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "conversation_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {

                            "intent": {
                                "type": "string",
                                "enum": [
                                    "greeting",
                                    "faq_question",
                                    "field_answer",
                                    "program_interest",
                                    "registration_request",
                                    "state_question",
                                    "permission_question",
                                    "human_agent_request",
                                    "complaint",
                                    "safety_risk",
                                    "meta_question",
                                    "pause_request",
                                    "unrelated",
                                ],
                            },

                            "questions": {
                                "type": "array",
                                "items": {
                                    "type": "string"
                                },
                            },

                            "parent_name": {
                                "type": "string"
                            },

                            "child_name": {
                                "type": "string"
                            },

                            "child_age": {
                                "type": "integer"
                            },

                            "main_concern": {
                                "type": "string"
                            },

                            "phone": {
                                "type": "string"
                            },

                            "preferred_call_time": {
                                "type": "string"
                            },

                            "multiple_children": {
                                "type": "boolean"
                            },

                            "children_count": {
                                "type": "integer"
                            },

                            "corrections": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "field": {
                                            "type": "string"
                                        },
                                        "value": {
                                            "type": "string"
                                        },
                                        "child_index": {
                                            "type": "integer"
                                        },
                                    },
                                    "required": [
                                        "field",
                                        "value",
                                        "child_index",
                                    ],
                                    "additionalProperties": False,
                                },
                            },

                            "confidence": {
                                "type": "number"
                            },
                        },

                        "required": [
                            "intent",
                            "questions",
                            "parent_name",
                            "child_name",
                            "child_age",
                            "main_concern",
                            "phone",
                            "preferred_call_time",
                            "multiple_children",
                            "children_count",
                            "corrections",
                            "confidence",
                        ],

                        "additionalProperties": False,
                    },
                },
            },
        )

        return json.loads(
            response.choices[0].message.content
        )

    except Exception as exc:

        print(
            "ANALYSIS ERROR:",
            exc
        )

        return {
            "intent": (
                "greeting"
                if is_greeting(user_text)
                else "field_answer"
            ),
            "questions": (
                [user_text]
                if "?" in user_text
                else []
            ),
            "parent_name": "",
            "child_name": "",
            "child_age": 0,
            "main_concern": "",
            "phone": "",
            "preferred_call_time": "",
            "multiple_children": False,
            "children_count": 0,
            "corrections": [],
            "confidence": 0.0,
        }


# =========================================================
# 12. APPLY CORRECTIONS
# =========================================================

def apply_corrections(
    lead: dict,
    corrections: List[dict],
):

    ensure_lead_structure(lead)

    for correction in corrections:

        field = correction.get(
            "field"
        )

        value = correction.get(
            "value",
            ""
        ).strip()

        child_index = correction.get(
            "child_index",
            0
        )

        if not value:
            continue

        if field == "parent_name":

            cleaned = clean_name(value)

            if cleaned:

                lead[
                    "parent_name"
                ] = cleaned

                lead[
                    "parent_title"
                ] = infer_parent_title_with_llm(
                    cleaned
                )

        elif field == "phone":

            phone = normalize_phone(value)

            if phone:
                lead["phone"] = phone

        elif field == "preferred_call_time":

            lead[
                "preferred_call_time"
            ] = value

        elif field in [
            "child_name",
            "child_age",
            "main_concern",
        ]:

            while len(
                lead["children"]
            ) <= child_index:

                lead["children"].append(
                    create_empty_child()
                )

            child = lead[
                "children"
            ][child_index]

            if field == "child_name":

                cleaned = clean_name(value)

                if cleaned:
                    child[
                        "name"
                    ] = cleaned

            elif field == "child_age":

                ages = extract_all_ages(
                    value
                )

                if ages:
                    child[
                        "age"
                    ] = ages[0]

            elif field == "main_concern":

                child[
                    "main_concern"
                ] = value

    sync_flat_fields(lead)


# =========================================================
# 13. MERGE EXTRACTED INFO
# =========================================================

def merge_analysis(
    lead: dict,
    analysis: dict,
    user_text: str,
):

    ensure_lead_structure(lead)

    # Corrections first
    apply_corrections(
        lead,
        analysis.get(
            "corrections",
            []
        )
    )

    deterministic_name = (
        deterministic_parent_name_extract(
            user_text
        )
    )

    parent_name = analysis.get(
        "parent_name",
        ""
    ).strip()

    if deterministic_name:
        parent_name = deterministic_name

    if (
        parent_name
        and not lead.get(
            "parent_name"
        )
    ):

        cleaned = clean_name(
            parent_name
        )

        if cleaned:

            lead[
                "parent_name"
            ] = cleaned

            lead[
                "parent_title"
            ] = infer_parent_title_with_llm(
                cleaned
            )

    # Multi child
    children_count = analysis.get(
        "children_count",
        0
    )

    if (
        analysis.get(
            "multiple_children"
        )
        and children_count >= 2
    ):

        lead[
            "multiple_children"
        ] = True

        while len(
            lead["children"]
        ) < children_count:

            lead["children"].append(
                create_empty_child()
            )

    ages = extract_all_ages(
        user_text
    )

    normalized_user = normalize_for_search(
        user_text
    )

    if (
        len(ages) >= 2
        and any(
            word in normalized_user
            for word in [
                "usaq",
                "ovlad",
            ]
        )
    ):

        lead[
            "multiple_children"
        ] = True

        while len(
            lead["children"]
        ) < len(ages):

            lead["children"].append(
                create_empty_child()
            )

        for i, age in enumerate(
            ages
        ):

            if not lead[
                "children"
            ][i].get(
                "age"
            ):

                lead[
                    "children"
                ][i][
                    "age"
                ] = age

    child = get_active_child(
        lead
    )

    child_name = analysis.get(
        "child_name",
        ""
    ).strip()

    if (
        child_name
        and not child.get(
            "name"
        )
    ):

        cleaned = clean_name(
            child_name
        )

        if cleaned:

            child[
                "name"
            ] = cleaned

    child_age = analysis.get(
        "child_age",
        0
    )

    if (
        child_age
        and not child.get(
            "age"
        )
    ):

        if 1 <= child_age <= 99:

            child[
                "age"
            ] = child_age

    concern = analysis.get(
        "main_concern",
        ""
    ).strip()

    if (
        concern
        and not child.get(
            "main_concern"
        )
    ):

        normalized_concern = normalize_for_search(
            concern
        )

        if normalized_concern in {
            "hamisi",
            "her biri",
            "hamisi",
        }:

            concern = (
                "özgüvən, məqsəd və gələcək, "
                "məsuliyyət və intizam, ünsiyyət"
            )

        child[
            "main_concern"
        ] = concern

        if any(
            word in normalize_for_search(
                concern
            )
            for word in [
                "fikirli",
                "ozune qapan",
                "danismir",
            ]
        ):

            child[
                "needs_concern_followup"
            ] = True

    extracted_phone = analysis.get(
        "phone",
        ""
    )

    phone = (
        normalize_phone(
            extracted_phone
        )
        or normalize_phone(
            user_text
        )
    )

    if (
        phone
        and not lead.get(
            "phone"
        )
    ):

        lead[
            "phone"
        ] = phone

    call_time = analysis.get(
        "preferred_call_time",
        ""
    ).strip()

    if (
        call_time
        and not lead.get(
            "preferred_call_time"
        )
    ):

        lead[
            "preferred_call_time"
        ] = call_time

    sync_flat_fields(lead)


# =========================================================
# 14. NEXT FIELD
# =========================================================

def child_is_complete(
    child: dict
) -> bool:

    if not child.get("name"):
        return False

    if not child.get("age"):
        return False

    if not child.get("main_concern"):
        return False

    if child.get(
        "needs_concern_followup"
    ):

        if not child.get(
            "concern_duration"
        ):
            return False

        if not child.get(
            "concern_onset"
        ):
            return False

    return True


def advance_child_if_needed(
    lead: dict
):

    ensure_lead_structure(lead)

    index = lead.get(
        "active_child_index",
        0
    )

    child = lead[
        "children"
    ][index]

    if not child_is_complete(
        child
    ):
        return

    for i, other_child in enumerate(
        lead["children"]
    ):

        if not child_is_complete(
            other_child
        ):

            lead[
                "active_child_index"
            ] = i

            return


def get_next_missing_field(
    lead: dict
):

    ensure_lead_structure(lead)

    if not lead.get(
        "parent_name"
    ):
        return "parent_name"

    advance_child_if_needed(
        lead
    )

    child = get_active_child(
        lead
    )

    if not child.get(
        "name"
    ):
        return "child_name"

    if not child.get(
        "age"
    ):
        return "child_age"

    if not child.get(
        "main_concern"
    ):
        return "main_concern"

    if child.get(
        "needs_concern_followup"
    ):

        if not child.get(
            "concern_duration"
        ):
            return "concern_duration"

        if not child.get(
            "concern_onset"
        ):
            return "concern_onset"

    for i, other_child in enumerate(
        lead["children"]
    ):

        if not child_is_complete(
            other_child
        ):

            lead[
                "active_child_index"
            ] = i

            return get_next_missing_field(
                lead
            )

    if not lead.get(
        "phone"
    ):
        return "phone"

    if not lead.get(
        "preferred_call_time"
    ):
        return "preferred_call_time"

    return None


# =========================================================
# 15. NEXT QUESTION
# =========================================================

def get_next_question(
    lead: dict
) -> str:

    field = get_next_missing_field(
        lead
    )

    child = get_active_child(
        lead
    )

    parent = get_parent_display_name(
        lead
    )

    if field == "parent_name":

        return (
            "Sizə necə müraciət edə bilərəm?"
        )

    if field == "child_name":

        if lead.get(
            "multiple_children"
        ):

            index = (
                lead.get(
                    "active_child_index",
                    0
                )
                + 1
            )

            return (
                f"{index}-ci övladınızın adını "
                "öyrənə bilərəm?"
            )

        if parent:

            return (
                f"Məmnun oldum, {parent}. "
                "Övladınızın adını öyrənə bilərəm?"
            )

        return (
            "Övladınızın adını öyrənə bilərəm?"
        )

    if field == "child_age":

        name = child.get(
            "name"
        )

        if name:

            return (
                f"{child_genitive(name)} neçə yaşı var?"
            )

        return (
            "Övladınızın neçə yaşı var?"
        )

    if field == "main_concern":

        return (
            "Ən çox hansı sahədə inkişaf etməsini istərdiniz?\n\n"
            "Məsələn: özgüvən, məqsəd və gələcək, "
            "məsuliyyət və intizam, ünsiyyət və s."
        )

    if field == "concern_duration":

        return (
            "Bu hal nə qədər müddətdir davam edir?"
        )

    if field == "concern_onset":

        return (
            "Sizcə bu vəziyyət hansısa hadisədən sonra "
            "başlayıb, yoxsa tədricən?"
        )

    if field == "phone":

        return (
            "Sizinlə əlaqə saxlaya bilməyimiz üçün "
            "telefon nömrənizi qeyd edin, zəhmət olmasa."
        )

    if field == "preferred_call_time":

        return (
            "Zəng üçün sizə hansı gün və saat aralığı "
            "daha uyğun olar?"
        )

    return ""


# =========================================================
# 16. HAS LEAD INFO
# =========================================================

def has_any_lead_info(
    lead: dict
) -> bool:

    ensure_lead_structure(
        lead
    )

    if lead.get(
        "parent_name"
    ):
        return True

    if lead.get(
        "phone"
    ):
        return True

    if lead.get(
        "preferred_call_time"
    ):
        return True

    for child in lead.get(
        "children",
        []
    ):

        if any([
            child.get("name"),
            child.get("age"),
            child.get("main_concern"),
        ]):
            return True

    return False


# =========================================================
# 17. STATE QUESTIONS
# =========================================================

def answer_state_question(
    user_text: str,
    lead: dict,
) -> Optional[str]:

    field = is_state_question(
        user_text
    )

    if not field:
        return None

    if field == "parent_name":

        parent = get_parent_display_name(
            lead
        )

        if parent:

            return (
                f"Bəli, adınızı {parent} kimi "
                "qeyd etmişəm. 😊"
            )

        return (
            "Hələ adınızı qeyd etməmişəm."
        )

    if field == "child_name":

        child = get_active_child(
            lead
        )

        name = child.get(
            "name"
        )

        if name:

            return (
                f"Bəli, övladınızın adını "
                f"{name} kimi qeyd etmişəm."
            )

        return (
            "Hələ övladınızın adını qeyd etməmişəm."
        )

    if field == "child_age":

        child = get_active_child(
            lead
        )

        age = child.get(
            "age"
        )

        if age:

            return (
                f"Bəli, övladınızın yaşını "
                f"{age} olaraq qeyd etmişəm."
            )

        return (
            "Hələ övladınızın yaşını qeyd etməmişəm."
        )

    if field == "phone":

        phone = lead.get(
            "phone"
        )

        if phone:

            return (
                f"Bəli, telefon nömrənizi "
                f"{phone} kimi qeyd etmişəm."
            )

        return (
            "Hələ telefon nömrənizi qeyd etməmişəm."
        )

    return None


# =========================================================
# 18. ANSWER QUESTIONS
# =========================================================

def answer_single_question(
    question: str,
    lead: dict,
    history: Optional[List[dict]] = None,
) -> str:

    state_answer = answer_state_question(
        question,
        lead,
    )

    if state_answer:

        return state_answer

    if is_permission_to_ask(
        question
    ):

        return (
            "Əlbəttə, buyurun 😊"
        )

    if is_child_presence_question(
        question
    ):

        return (
            "İlkin zəng zamanı övladınızın "
            "iştirakı vacib deyil."
        )

    if is_contact_here_question(
        question
    ):

        return (
            "Bəli 😊 Buradan müraciətinizi qeyd edə bilərsiniz. "
            "Məlumatlar tamamlandıqdan sonra Junior Coaching "
            "komandası sizinlə əlaqə saxlayacaq."
        )

    if is_bot_question(
        question
    ):

        return (
            "Mən Junior Coaching proqramı üzrə "
            "virtual müraciət köməkçisiyəm 😊"
        )

    candidates = retrieve_faq_candidates(
        question,
        k=5,
    )

    faq = select_best_faq_with_llm(
        question,
        candidates,
    )

    if faq:

        lead[
            "_last_faq_score"
        ] = faq.get(
            "score"
        )

        return faq[
            "answer"
        ]

    return (
        "Bu sualla bağlı məlumat bazasında "
        "dəqiq cavab tapmadım. "
        "İstəsəniz bu sualı məsul əməkdaşa "
        "yönləndirə bilərik."
    )


def naturalize_repeated_answer(
    user_text: str,
    answer: str,
    history: Optional[List[dict]] = None,
) -> str:

    """
    Eyni mövzuda user ikinci dəfə israr edirsə,
    eyni FAQ cavabını sözbəsöz təkrar etməməyə çalışır.
    """

    history = history or []

    if not history:
        return answer

    recent_assistant_answers = [
        item.get(
            "content",
            ""
        )
        for item in history[-6:]
        if item.get(
            "role"
        ) == "assistant"
    ]

    if answer not in recent_assistant_answers:
        return answer

    if client is None:

        return (
            "Sizi başa düşürəm. "
            + answer
        )

    try:

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,

            messages=[
                {
                    "role": "system",
                    "content": """
İstifadəçiyə verilən əvvəlki faktiki cavabı dəyişmədən,
amma eyni cümləni sözbəsöz təkrarlamadan daha təbii follow-up yaz.

Yeni fakt uydurma.
Qiymət rəqəmi source-da yoxdursa rəqəm uydurma.
Qısa və empatik ol.
"""
                },
                {
                    "role": "user",
                    "content": (
                        f"User follow-up:\n{user_text}\n\n"
                        f"Əvvəlki faktiki cavab:\n{answer}"
                    ),
                },
            ],
        )

        text = response.choices[
            0
        ].message.content.strip()

        if text:
            return text

    except Exception as exc:

        print(
            "REPEAT NATURALIZE ERROR:",
            exc
        )

    return answer


def answer_user_questions(
    user_text: str,
    questions: List[str],
    lead: dict,
    history: Optional[List[dict]] = None,
) -> str:

    answers = []

    for question in questions:

        answer = answer_single_question(
            question,
            lead,
            history,
        )

        answer = naturalize_repeated_answer(
            user_text,
            answer,
            history,
        )

        if (
            answer
            and answer not in answers
        ):
            answers.append(answer)

    if not answers:

        return (
            "Bu sualla bağlı dəqiq cavab tapa bilmədim."
        )

    return "\n\n".join(
        answers
    )


# =========================================================
# 19. FALLBACK FIELD SAVE
# =========================================================

def save_current_field_fallback(
    lead: dict,
    field: str,
    user_text: str,
):

    child = get_active_child(
        lead
    )

    value = user_text.strip()

    normalized = normalize_for_search(
        value
    )

    if field == "parent_name":

        name = (
            deterministic_parent_name_extract(
                value
            )
            or clean_name(
                remove_honorific(
                    value
                )
            )
        )

        if name:

            lead[
                "parent_name"
            ] = name

            lead[
                "parent_title"
            ] = infer_parent_title_with_llm(
                name
            )

    elif field == "child_name":

        name = clean_name(
            value
        )

        if name:

            child[
                "name"
            ] = name

    elif field == "child_age":

        ages = extract_all_ages(
            value
        )

        if len(ages) == 1:

            child[
                "age"
            ] = ages[0]

    elif field == "main_concern":

        if normalized in {
            "hamisi",
            "her biri",
        }:

            value = (
                "özgüvən, məqsəd və gələcək, "
                "məsuliyyət və intizam, ünsiyyət"
            )

        child[
            "main_concern"
        ] = value

        if any(
            x in normalized
            for x in [
                "fikirli",
                "ozune qapan",
                "danismir",
            ]
        ):

            child[
                "needs_concern_followup"
            ] = True

    elif field == "concern_duration":

        child[
            "concern_duration"
        ] = value

    elif field == "concern_onset":

        child[
            "concern_onset"
        ] = value

    elif field == "phone":

        phone = normalize_phone(
            value
        )

        if phone:

            lead[
                "phone"
            ] = phone

    elif field == "preferred_call_time":

        # Tək "sabah"ı final vaxt saymayaq
        if normalized not in {
            "sabah",
            "bugun",
            "bu gun",
        }:

            lead[
                "preferred_call_time"
            ] = value

    sync_flat_fields(
        lead
    )


# =========================================================
# 20. FINAL MESSAGE
# =========================================================

def build_final_message(
    lead: dict
) -> str:

    parent = get_parent_display_name(
        lead
    )

    call_time = lead.get(
        "preferred_call_time"
    )

    result = "Qeydə alındı ✅"

    if parent and call_time:

        result += (
            f"\n\n{parent}, {call_time} "
            "sizinlə əlaqə saxlanılması üçün "
            "müraciətinizi qeyd etdim."
        )

    elif call_time:

        result += (
            f"\n\n{call_time} sizinlə əlaqə "
            "saxlanılması üçün müraciətinizi qeyd etdim."
        )

    result += (
        "\n\nİlkin zəng zamanı övladınızın "
        "iştirakı vacib deyil."
    )

    return result


# =========================================================
# 21. MAIN AGENT
# =========================================================

def lead_agent_reply(
    user_text: str,
    lead: dict,
    faq_min_score: float = 0.18,
    history: Optional[List[dict]] = None,
) -> str:

    user_text = user_text.strip()

    history = history or []

    ensure_lead_structure(
        lead
    )

    field_before = get_next_missing_field(
        lead
    )

    lead[
        "_last_faq_score"
    ] = None

    # =====================================================
    # 1. Analyze whole message + conversation
    # =====================================================

    analysis = analyze_message(
        user_text=user_text,
        lead=lead,
        history=history,
    )

    print(
        "ANALYSIS DEBUG:",
        analysis,
    )

    lead[
        "_last_intent"
    ] = analysis.get(
        "intent"
    )

    lead[
        "_last_confidence"
    ] = analysis.get(
        "confidence"
    )

    # =====================================================
    # 2. Merge all information and corrections
    # =====================================================

    merge_analysis(
        lead=lead,
        analysis=analysis,
        user_text=user_text,
    )

    # =====================================================
    # 3. Safety
    # =====================================================

    if analysis.get(
        "intent"
    ) == "safety_risk":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Bu vəziyyət peşəkar və təcili diqqət "
            "tələb edə bilər. Junior Coaching tibbi və "
            "ya psixoloji təcili yardımı əvəz etmir. "
            "Müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # =====================================================
    # 4. Human handoff
    # =====================================================

    if analysis.get(
        "intent"
    ) == "human_agent_request":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Əlbəttə. Müraciətinizi Junior Coaching "
            "komandasından məsul əməkdaşa yönləndirmək "
            "üçün qeydə aldım."
        )

    # =====================================================
    # 5. Complaint
    # =====================================================

    if analysis.get(
        "intent"
    ) == "complaint":

        # Uşağın problemi complaint deyil
        if field_before != "main_concern":

            lead[
                "status"
            ] = "ESCALATED"

            return (
                "Narahatlığınızı başa düşürəm. "
                "Müraciətinizi məsul əməkdaşa "
                "yönləndirmək üçün qeydə aldım."
            )

    # =====================================================
    # 6. Permission to ask
    # =====================================================

    if (
        analysis.get(
            "intent"
        ) == "permission_question"
        or is_permission_to_ask(
            user_text
        )
    ):

        return (
            "Əlbəttə, buyurun 😊"
        )

    # =====================================================
    # 7. State question
    # =====================================================

    if (
        analysis.get(
            "intent"
        ) == "state_question"
        or is_state_question(
            user_text
        )
    ):

        state_answer = answer_state_question(
            user_text,
            lead,
        )

        if state_answer:
            return state_answer

    # =====================================================
    # 8. Pure greeting
    # =====================================================

    if (
        analysis.get(
            "intent"
        ) == "greeting"
        and is_greeting(
            user_text
        )
    ):

        if not lead.get(
            "_greeted"
        ):

            lead[
                "_greeted"
            ] = True

            return (
                "Salam 😊\n\n"
                + get_next_question(
                    lead
                )
            )

        # artıq salamlaşmışıq
        question = get_next_question(
            lead
        )

        if question:

            return question

        return (
            "Buyurun 😊"
        )

    # =====================================================
    # 9. Questions first
    # =====================================================

    questions = analysis.get(
        "questions",
        []
    )

    if not questions:

        # deterministic question fallback
        if (
            "?" in user_text
            or analysis.get(
                "intent"
            ) in [
                "faq_question",
                "program_interest",
                "meta_question",
            ]
        ):

            questions = [
                user_text
            ]

    if questions:

        answer = answer_user_questions(
            user_text=user_text,
            questions=questions,
            lead=lead,
            history=history,
        )

        # ---------------------------------------------
        # CONVERSATION BRIDGE
        # ---------------------------------------------

        has_lead_info = has_any_lead_info(
            lead
        )

        if not has_lead_info:

            answer += (
                "\n\nBaşqa sualınız varsa, buyurun 😊 "
                "Müraciət etmək istəyirsinizsə, "
                "sizə necə müraciət edə bilərəm?"
            )

            return answer

        # artıq müraciət məlumatı verilib
        # yalnız 1 next-flow question
        next_question = get_next_question(
            lead
        )

        if next_question:

            answer += (
                "\n\n"
                + next_question
            )

        return answer

    # =====================================================
    # 10. If extractor did not advance current field,
    # use fallback
    # =====================================================

    field_after_merge = get_next_missing_field(
        lead
    )

    if (
        field_before
        and field_before == field_after_merge
    ):

        save_current_field_fallback(
            lead=lead,
            field=field_before,
            user_text=user_text,
        )

    # =====================================================
    # 11. Completion
    # =====================================================

    next_field = get_next_missing_field(
        lead
    )

    if next_field is None:

        lead[
            "status"
        ] = "CALL_REQUESTED"

        return build_final_message(
            lead
        )

    # =====================================================
    # 12. Flow has started after first information
    # =====================================================

    if has_any_lead_info(
        lead
    ):

        lead[
            "_flow_started"
        ] = True

    # =====================================================
    # 13. Maximum one next question
    # =====================================================

    return get_next_question(
        lead
    )


# =========================================================
# 22. DATABASE INIT
# =========================================================

def init_db():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS leads (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                parent_name TEXT,
                parent_title TEXT,

                child_name TEXT,
                child_age INTEGER,
                main_concern TEXT,

                needs_concern_followup INTEGER DEFAULT 0,
                concern_duration TEXT,
                concern_onset TEXT,

                children_json TEXT,

                phone TEXT,
                preferred_call_time TEXT,

                source TEXT,
                status TEXT,

                created_at TEXT,
                updated_at TEXT
            )
            """
        )

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_logs (

                id INTEGER PRIMARY KEY AUTOINCREMENT,

                session_id TEXT NOT NULL,

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

                children_json TEXT,

                phone TEXT,
                preferred_call_time TEXT,

                status TEXT,
                source TEXT,

                created_at TEXT
            )
            """
        )

        existing = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(leads)"
            ).fetchall()
        }

        required = {
            "parent_title": "TEXT",
            "needs_concern_followup":
                "INTEGER DEFAULT 0",
            "concern_duration": "TEXT",
            "concern_onset": "TEXT",
            "children_json": "TEXT",
        }

        for column, dtype in required.items():

            if column not in existing:

                conn.execute(
                    f"""
                    ALTER TABLE leads
                    ADD COLUMN {column} {dtype}
                    """
                )

        existing_logs = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(conversation_logs)"
            ).fetchall()
        }

        log_required = {
            "intent": "TEXT",
            "confidence": "REAL",
            "faq_score": "REAL",
            "parent_title": "TEXT",
            "child_age": "INTEGER",
            "main_concern": "TEXT",
            "children_json": "TEXT",
            "preferred_call_time": "TEXT",
            "source": "TEXT",
        }

        for column, dtype in log_required.items():

            if column not in existing_logs:

                conn.execute(
                    f"""
                    ALTER TABLE conversation_logs
                    ADD COLUMN {column} {dtype}
                    """
                )

        conn.commit()


# =========================================================
# 23. TIME
# =========================================================

def get_baku_time():

    return datetime.now(
        ZoneInfo(
            "Asia/Baku"
        )
    ).strftime(
        "%Y-%m-%d %H:%M:%S"
    )


# =========================================================
# 24. FIND LEAD
# =========================================================

def find_lead_by_phone(
    phone: str
):

    if not phone:
        return None

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        row = conn.execute(
            """
            SELECT *
            FROM leads
            WHERE phone = ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                phone,
            ),
        ).fetchone()

        if row:
            return dict(row)

    return None


# =========================================================
# 25. SAVE LEAD
# =========================================================

def save_lead_to_db(
    lead: dict
) -> int:

    ensure_lead_structure(
        lead
    )

    sync_flat_fields(
        lead
    )

    now = get_baku_time()

    children_json = json.dumps(
        lead.get(
            "children",
            []
        ),
        ensure_ascii=False,
    )

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO leads (

                parent_name,
                parent_title,

                child_name,
                child_age,
                main_concern,

                needs_concern_followup,
                concern_duration,
                concern_onset,

                children_json,

                phone,
                preferred_call_time,

                source,
                status,

                created_at,
                updated_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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

                int(
                    bool(
                        lead.get(
                            "needs_concern_followup",
                            False
                        )
                    )
                ),

                lead.get(
                    "concern_duration"
                ),

                lead.get(
                    "concern_onset"
                ),

                children_json,

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

                now,
                now,
            ),
        )

        conn.commit()

        return cursor.lastrowid


# =========================================================
# 26. SAVE CONVERSATION LOG
# =========================================================

def save_conversation_log(
    session_id: str,
    user_message: str,
    bot_response: str,
    current_field: Optional[str],
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    sync_flat_fields(
        lead
    )

    now = get_baku_time()

    children_json = json.dumps(
        lead.get(
            "children",
            []
        ),
        ensure_ascii=False,
    )

    with sqlite3.connect(
        DB_PATH
    ) as conn:

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

                children_json,

                phone,
                preferred_call_time,

                status,
                source,

                created_at

            )

            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                session_id,

                user_message,
                bot_response,

                lead.get(
                    "_last_intent"
                ),

                lead.get(
                    "_last_confidence"
                ),

                lead.get(
                    "_last_faq_score"
                ),

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

                children_json,

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

                now,
            ),
        )

        conn.commit()


# =========================================================
# 27. ADMIN
# =========================================================

def get_all_leads():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT *
            FROM leads
            ORDER BY id DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


def get_all_conversation_logs():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = sqlite3.Row

        rows = conn.execute(
            """
            SELECT *
            FROM conversation_logs
            ORDER BY id DESC
            """
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]


# =========================================================
# INIT
# =========================================================

init_db()