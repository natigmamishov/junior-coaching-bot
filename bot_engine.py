"""
Junior Coaching — Bot Engine V3

Əsas prinsiplər
---------------
1. Hər user mesajı bütöv analiz olunur.
2. Bir mesajdan bir neçə məlumat eyni anda çıxarıla bilər.
3. Artıq məlum olan məlumat yenidən soruşulmur.
4. User arada sual verirsə:
      əvvəl suala cavab verilir,
      sonra maksimum 1 çatışmayan məlumat soruşulur.
5. main_concern mərhələsində:
      "məsuliyyətsizdir",
      "özgüvəni zəifdir",
      "məqsəd və gələcək",
      "hamısı"
   complaint / FAQ kimi qəbul edilmir.
6. İki və daha çox uşaq üçün children strukturu saxlanılır.
7. SQLite + Streamlit app.py ilə backward compatible-dir.
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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "Junior_Coaching_sesli_AI_FAQ.txt",
)

DB_PATH = os.path.join(
    BASE_DIR,
    "junior_coaching.db",
)


# =========================================================
# 1. OPENAI
# =========================================================

load_dotenv(
    os.path.join(
        BASE_DIR,
        ".env",
    )
)

OPENAI_API_KEY = os.getenv(
    "OPENAI_API_KEY"
)

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


def normalize_text(
    text: str,
) -> str:

    return re.sub(
        r"\s+",
        " ",
        str(text).strip().lower(),
    )


def normalize_for_search(
    text: str,
) -> str:

    text = normalize_text(
        text
    )

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
# 3. GENERAL HELPERS
# =========================================================

def normalize_phone(
    text: str,
) -> Optional[str]:

    digits = re.sub(
        r"\D",
        "",
        text,
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


def extract_all_ages(
    text: str,
) -> List[int]:

    result = []

    for value in re.findall(
        r"\b\d{1,2}\b",
        text,
    ):

        number = int(
            value
        )

        if (
            1 <= number <= 99
            and number not in result
        ):
            result.append(
                number
            )

    return result


def is_greeting(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

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

def remove_honorific(
    text: str,
) -> str:

    words = text.strip().split()

    blocked = {
        "bey",
        "xanim",
        "muellim",
    }

    result = []

    for word in words:

        if normalize_for_search(
            word
        ) not in blocked:

            result.append(
                word
            )

    return " ".join(
        result
    )


def remove_name_suffix(
    word: str,
) -> str:

    """
    ismayildir -> ismayil
    orxandir   -> orxan
    elvindir   -> elvin

    Amma Nadir -> Nadir qalır.
    """

    if not word:
        return word

    normalized = normalize_for_search(
        word
    )

    for suffix in [
        "dir",
        "dur",
    ]:

        if normalized.endswith(
            suffix
        ):

            base_length = (
                len(word)
                - len(suffix)
            )

            if base_length >= 4:

                return word[
                    :base_length
                ]

    return word


def clean_name(
    value: str,
) -> Optional[str]:

    if not value:
        return None

    value = remove_honorific(
        value
    )

    value = re.sub(
        r"[.,!?+]",
        "",
        value,
    ).strip()

    if not value:
        return None

    words = []

    for word in value.split():

        cleaned = remove_name_suffix(
            word
        )

        if cleaned:
            words.append(
                cleaned
            )

    if not words:
        return None

    # Maksimum 2 söz
    value = " ".join(
        words[:2]
    )

    if not re.fullmatch(
        r"[A-Za-zƏəÖöÜüĞğÇçŞşİı\- ]+",
        value,
    ):
        return None

    return value.title()


def deterministic_name_extract(
    text: str,
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
# 5. AZERBAIJANI SUFFIXES
# =========================================================

def get_last_vowel(
    word: str,
) -> Optional[str]:

    vowels = "aıoueəiöü"

    for char in reversed(
        word.lower()
    ):

        if char in vowels:
            return char

    return None


def get_genitive_suffix(
    word: str,
) -> str:

    vowel = get_last_vowel(
        word
    )

    if vowel in [
        "a",
        "ı",
    ]:
        return "ın"

    if vowel in [
        "e",
        "ə",
        "i",
    ]:
        return "in"

    if vowel in [
        "o",
        "u",
    ]:
        return "un"

    if vowel in [
        "ö",
        "ü",
    ]:
        return "ün"

    return "ın"


def child_genitive(
    name: str,
) -> str:

    if not name:
        return "övladınızın"

    suffix = get_genitive_suffix(
        name
    )

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
    source: str = "CLI",
) -> dict:

    return {

        "parent_name": None,
        "parent_title": None,

        # Backward compatibility
        "child_name": None,
        "child_age": None,
        "main_concern": None,

        "needs_concern_followup": False,
        "concern_duration": None,
        "concern_onset": None,

        # V3
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
    }


def ensure_lead_structure(
    lead: dict,
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
        0,
    )

    lead.setdefault(
        "multiple_children",
        False,
    )


def get_active_child(
    lead: dict,
) -> dict:

    ensure_lead_structure(
        lead
    )

    index = lead.get(
        "active_child_index",
        0,
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
    lead: dict,
):

    """
    app.py və köhnə DB strukturu üçün
    birinci uşağın məlumatlarını flat saxlayırıq.
    """

    ensure_lead_structure(
        lead
    )

    first = lead[
        "children"
    ][0]

    lead[
        "child_name"
    ] = first.get(
        "name"
    )

    lead[
        "child_age"
    ] = first.get(
        "age"
    )

    lead[
        "main_concern"
    ] = first.get(
        "main_concern"
    )

    lead[
        "needs_concern_followup"
    ] = first.get(
        "needs_concern_followup",
        False,
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
# 7. PARENT TITLE
# =========================================================

def infer_parent_title_with_llm(
    parent_name: str,
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

Əmin deyilsənsə neutral.
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

        data = json.loads(
            response.choices[0].message.content
        )

        if data["title"] in [
            "xanım",
            "bəy",
        ]:

            return data[
                "title"
            ]

    except Exception as exc:

        print(
            "TITLE ERROR:",
            exc,
        )

    return ""


def get_parent_display_name(
    lead: dict,
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
    ) as f:

        raw = f.read()

    pattern = re.compile(
        r"(?:\d+\.\s*)?Sual:\s*(.*?)\s*"
        r"(?:Agent|Selnaz|Cavab):\s*(.*?)"
        r"(?=\n\s*(?:\d+\.\s*)?Sual:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    pairs = []

    for q, a in pattern.findall(
        raw
    ):

        q = re.sub(
            r"\s+",
            " ",
            q,
        ).strip()

        a = re.sub(
            r"\s+",
            " ",
            a,
        ).strip()

        if q and a:

            pairs.append(
                (
                    q,
                    a,
                )
            )

    questions = [
        q
        for q, _ in pairs
    ]

    answers = [
        a
        for _, a in pairs
    ]

    normalized = [
        normalize_for_search(
            q
        )
        for q in questions
    ]

    vectorizer = FeatureUnion([

        (
            "word",
            TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                sublinear_tf=True,
            ),
        ),

        (
            "char",
            TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(3, 5),
                sublinear_tf=True,
            ),
        ),

    ])

    matrix = vectorizer.fit_transform(
        normalized
    )

    return (
        questions,
        answers,
        vectorizer,
        matrix,
    )


FAQ_QUESTIONS, FAQ_ANSWERS, FAQ_VECTORIZER, FAQ_MATRIX = (
    build_faq_index()
)


def answer_faq_question(
    user_text: str,
    min_score: float = 0.18,
):

    query = normalize_for_search(
        user_text
    )

    vector = FAQ_VECTORIZER.transform(
        [query]
    )

    scores = cosine_similarity(
        vector,
        FAQ_MATRIX,
    ).ravel()

    best_index = int(
        np.argmax(scores)
    )

    best_score = float(
        scores[
            best_index
        ]
    )

    if best_score < min_score:

        return None

    return {
        "question": FAQ_QUESTIONS[
            best_index
        ],
        "answer": FAQ_ANSWERS[
            best_index
        ],
        "score": best_score,
    }


# =========================================================
# 9. SPECIAL QUESTION DETECTORS
# =========================================================

def is_child_presence_question(
    text: str,
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
    text: str,
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
    text: str,
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
        x in value
        for x in patterns
    )


def is_call_time_question(
    text: str,
) -> bool:

    value = normalize_for_search(
        text
    )

    return (
        any(
            x in value
            for x in [
                "sabah",
                "bugun",
                "axsam",
                "seher",
                "gunorta",
                "heftesonu",
            ]
        )
        and any(
            x in value
            for x in [
                "zeng",
                "elaqe",
                "danismaq",
                "mumkundur",
                "olar",
                "alinar",
            ]
        )
    )


# =========================================================
# 10. WHOLE MESSAGE EXTRACTION — V3 CORE
# =========================================================

def extract_message_with_llm(
    user_text: str,
    lead: dict,
) -> dict:

    """
    Bu V3-ün əsas hissəsidir.

    Bir mesajdan eyni anda bütün mümkün slotları çıxarır.
    """

    ensure_lead_structure(
        lead
    )

    state = {
        "parent_name": lead.get(
            "parent_name"
        ),
        "phone": lead.get(
            "phone"
        ),
        "preferred_call_time": lead.get(
            "preferred_call_time"
        ),
        "children": lead.get(
            "children"
        ),
    }

    if client is None:

        return {
            "intent": "field_answer",
            "is_question": False,
            "question_text": "",
            "parent_name": "",
            "child_name": "",
            "child_age": 0,
            "main_concern": "",
            "phone": "",
            "preferred_call_time": "",
            "multiple_children": False,
            "children_count": 0,
            "confidence": 0.0,
        }

    system_prompt = """
Sən Junior Coaching müraciət sisteminin message extraction modulusan.

Sənin vəzifən istifadəçinin BÜTÜN mesajını analiz etməkdir.

Bir mesajda bir neçə məlumat varsa HAMISINI çıxar.

Məsələn:

"Mən Nərgizəm. Oğlum Orxanın 15 yaşı var.
Özünəinamı zəifdir. Nömrəm 0501234567-dir.
Sabah 15:00-dan sonra danışa bilərəm."

Nəticə:
parent_name=Nərgiz
child_name=Orxan
child_age=15
main_concern=özünəinam
phone=0501234567
preferred_call_time=sabah 15:00-dan sonra


ÇOX VACİB:

1. "məsuliyyətsizdir", "özgüvəni zəifdir",
   "ünsiyyəti zəifdir", "məqsəd və gələcək",
   "futbolla məşğuldur", "hamısı", "hər biri"
   complaint DEYİL.

Bunlar valideynin övladı üçün ehtiyac/inkişaf sahəsidir.

2. Cari flow-a baxmayaraq mesajın bütün məlumatlarını çıxar.

3. Artıq state-də olan məlumatı user yenidən yazmayıbsa dəyişmə.

4. "Mən Aygünəm" -> parent_name = Aygün.
"Mən Aygün xanım" YOX.

5. "Adım İsmayıldır" -> parent_name = İsmayıl.

6. "adı Orxandır" -> child_name = Orxan.

7. Mesaj user-in real sualıdırsa is_question=true.

Məsələn:
"Qiymət nə qədərdir?"
"Rus dilli qrup var?"
"Uşaq zəngdə yanımda olmalıdır?"
"Nə üçün uşağın adını soruşursunuz?"

8. User sual verir və eyni zamanda məlumat verirsə,
həm sualı, həm məlumatları çıxar.

9. Bir neçə uşaq varsa multiple_children=true.

"13 və 15 yaşında iki uşağım var"
=> children_count=2

10. main_concern nümunələri:
özgüvən
məqsəd və gələcək
məsuliyyət
intizam
ünsiyyət
fokus
liderlik
emosional zəka
qərarvermə
dərs
telefon asılılığı
sosiallaşma

11. "hamısı", "hər biri" main_concern kimi qəbul edilə bilər.

Intent:
greeting
faq_question
field_answer
program_interest
registration_request
human_agent_request
complaint
safety_risk
meta_question
pause_request
unrelated

Uşağa aid mənfi xüsusiyyəti complaint kimi seçmə.
Complaint yalnız istifadəçi Junior Coaching xidmətindən narazılıq bildirəndə seçilməlidir.
"""

    user_prompt = f"""
CURRENT STATE:
{json.dumps(state, ensure_ascii=False)}

USER MESSAGE:
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
                    "content": user_prompt,
                },
            ],

            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "junior_message_extract",
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
                                    "human_agent_request",
                                    "complaint",
                                    "safety_risk",
                                    "meta_question",
                                    "pause_request",
                                    "unrelated",
                                ],
                            },

                            "is_question": {
                                "type": "boolean"
                            },

                            "question_text": {
                                "type": "string"
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

                            "confidence": {
                                "type": "number"
                            },
                        },

                        "required": [
                            "intent",
                            "is_question",
                            "question_text",
                            "parent_name",
                            "child_name",
                            "child_age",
                            "main_concern",
                            "phone",
                            "preferred_call_time",
                            "multiple_children",
                            "children_count",
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
            "MESSAGE EXTRACT ERROR:",
            exc,
        )

        return {
            "intent": (
                "greeting"
                if is_greeting(
                    user_text
                )
                else "field_answer"
            ),
            "is_question": (
                "?" in user_text
            ),
            "question_text": (
                user_text
                if "?" in user_text
                else ""
            ),
            "parent_name": "",
            "child_name": "",
            "child_age": 0,
            "main_concern": "",
            "phone": (
                normalize_phone(
                    user_text
                )
                or ""
            ),
            "preferred_call_time": "",
            "multiple_children": False,
            "children_count": 0,
            "confidence": 0.0,
        }


# =========================================================
# 11. MERGE EXTRACTED INFORMATION
# =========================================================

def merge_extracted_information(
    lead: dict,
    data: dict,
    user_text: str,
):

    """
    Yalnız boş slotları doldurur.
    Artıq məlum məlumatı təkrar overwrite etmir.
    """

    ensure_lead_structure(
        lead
    )

    # ---------------------------------------------
    # Parent name
    # ---------------------------------------------

    parent_name = data.get(
        "parent_name",
        "",
    ).strip()

    deterministic_name = (
        deterministic_name_extract(
            user_text
        )
    )

    if deterministic_name:

        parent_name = (
            deterministic_name
        )

    if (
        parent_name
        and not lead.get(
            "parent_name"
        )
    ):

        parent_name = clean_name(
            parent_name
        )

        if parent_name:

            lead[
                "parent_name"
            ] = parent_name

            lead[
                "parent_title"
            ] = infer_parent_title_with_llm(
                parent_name
            )

    # ---------------------------------------------
    # Multiple children
    # ---------------------------------------------

    children_count = data.get(
        "children_count",
        0,
    )

    if (
        data.get(
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

            lead[
                "children"
            ].append(
                create_empty_child()
            )

    # Deterministic multiple ages
    ages = extract_all_ages(
        user_text
    )

    if (
        len(ages) >= 2
        and any(
            x in normalize_for_search(
                user_text
            )
            for x in [
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

            lead[
                "children"
            ].append(
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

    # ---------------------------------------------
    # Active child
    # ---------------------------------------------

    child = get_active_child(
        lead
    )

    child_name = data.get(
        "child_name",
        "",
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

    child_age = data.get(
        "child_age",
        0,
    )

    if (
        child_age
        and not child.get(
            "age"
        )
    ):

        if (
            1 <= child_age <= 99
        ):

            child[
                "age"
            ] = child_age

    concern = data.get(
        "main_concern",
        "",
    ).strip()

    if (
        concern
        and not child.get(
            "main_concern"
        )
    ):

        normalized = normalize_for_search(
            concern
        )

        if normalized in {
            "hamisi",
            "her biri",
            "hamisi olsun",
            "her biri olsun",
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

    # ---------------------------------------------
    # Phone
    # ---------------------------------------------

    extracted_phone = data.get(
        "phone",
        "",
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

    # ---------------------------------------------
    # Call time
    # ---------------------------------------------

    call_time = data.get(
        "preferred_call_time",
        "",
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

    sync_flat_fields(
        lead
    )


# =========================================================
# 12. NEXT MISSING FIELD
# =========================================================

def child_is_complete(
    child: dict,
) -> bool:

    if not child.get(
        "name"
    ):
        return False

    if not child.get(
        "age"
    ):
        return False

    if not child.get(
        "main_concern"
    ):
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
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

    index = lead.get(
        "active_child_index",
        0,
    )

    current = lead[
        "children"
    ][index]

    if not child_is_complete(
        current
    ):
        return

    # Növbəti incomplete child
    for i, child in enumerate(
        lead["children"]
    ):

        if not child_is_complete(
            child
        ):

            lead[
                "active_child_index"
            ] = i

            return


def get_next_missing_field(
    lead: dict,
):

    ensure_lead_structure(
        lead
    )

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

    # Başqa incomplete uşaq?
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
# 13. ONE QUESTION ONLY
# =========================================================

def get_next_question(
    lead: dict,
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
                    0,
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
# 14. FIELD-SPECIFIC FALLBACK SAVE
# =========================================================

def save_current_field_fallback(
    lead: dict,
    field: str,
    user_text: str,
):

    """
    LLM slot çıxarmasa belə cari suala verilən
    sadə cavabı itirmirik.
    """

    child = get_active_child(
        lead
    )

    value = user_text.strip()

    normalized = normalize_for_search(
        value
    )

    if field == "parent_name":

        name = (
            deterministic_name_extract(
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

        if len(
            ages
        ) == 1:

            child[
                "age"
            ] = ages[0]

    elif field == "main_concern":

        # Bu sahədə complaint/FAQ kimi səhv classification etməmək üçün
        # qısa development cavabları birbaşa qəbul edilir.

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

        lead[
            "preferred_call_time"
        ] = value

    sync_flat_fields(
        lead
    )


# =========================================================
# 15. SPECIAL QUESTION ANSWERS
# =========================================================

def answer_special_question(
    user_text: str,
    lead: dict,
) -> Optional[str]:

    if is_child_presence_question(
        user_text
    ):

        return (
            "İlkin zəng zamanı övladınızın iştirakı vacib deyil."
        )

    if is_contact_here_question(
        user_text
    ):

        return (
            "Bəli 😊 Buradan müraciətinizi qeyd edə bilərsiniz. "
            "Məlumatlar tamamlandıqdan sonra Junior Coaching "
            "komandası sizinlə əlaqə saxlayacaq."
        )

    if is_call_time_question(
        user_text
    ):

        return (
            "Bəli, mümkündür 😊 Sizə uyğun gün və saat "
            "nəzərə alınaraq əlaqə saxlanıla bilər."
        )

    if is_bot_question(
        user_text
    ):

        return (
            "Mən Junior Coaching proqramı üzrə virtual "
            "müraciət köməkçisiyəm 😊"
        )

    return None


# =========================================================
# 16. FAQ ANSWER + ONE NEXT QUESTION
# =========================================================

def answer_user_question(
    user_text: str,
    lead: dict,
    faq_min_score: float,
) -> str:

    special = answer_special_question(
        user_text,
        lead,
    )

    if special:

        answer = special

    else:

        faq = answer_faq_question(
            user_text,
            min_score=faq_min_score,
        )

        if faq:

            lead[
                "_last_faq_score"
            ] = faq[
                "score"
            ]

            answer = faq[
                "answer"
            ]

        else:

            answer = (
                "Bu sualla bağlı məlumat bazasında "
                "dəqiq cavab tapmadım. "
                "İstəsəniz bu sualı məsul əməkdaşa "
                "yönləndirə bilərik."
            )

    # Maksimum 1 növbəti sual
    next_question = get_next_question(
        lead
    )

    if next_question:

        answer += (
            "\n\n"
            + next_question
        )

    return answer


# =========================================================
# 17. FINAL MESSAGE
# =========================================================

def build_final_message(
    lead: dict,
) -> str:

    parent = get_parent_display_name(
        lead
    )

    call_time = lead.get(
        "preferred_call_time"
    )

    if call_time:

        result = (
            f"Qeydə alındı ✅\n\n"
            f"{parent + ', ' if parent else ''}"
            f"{call_time} sizinlə əlaqə saxlanılması "
            f"üçün müraciətinizi qeyd etdim."
        )

    else:

        result = (
            "Qeydə alındı ✅"
        )

    result += (
        "\n\nİlkin zəng zamanı övladınızın "
        "iştirakı vacib deyil."
    )

    return result


# =========================================================
# 18. MAIN AGENT
# =========================================================

def lead_agent_reply(
    user_text: str,
    lead: dict,
    faq_min_score: float = 0.18,
) -> str:

    user_text = user_text.strip()

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
    # 1. Whole-message extraction
    # =====================================================

    data = extract_message_with_llm(
        user_text,
        lead,
    )

    print(
        "EXTRACT DEBUG:",
        data,
    )

    lead[
        "_last_intent"
    ] = data.get(
        "intent"
    )

    lead[
        "_last_confidence"
    ] = data.get(
        "confidence"
    )

    # =====================================================
    # 2. Merge ALL slots
    # =====================================================

    merge_extracted_information(
        lead,
        data,
        user_text,
    )

    # =====================================================
    # 3. Safety
    # =====================================================

    if data.get(
        "intent"
    ) == "safety_risk":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Bu vəziyyət peşəkar və təcili diqqət tələb edə bilər. "
            "Junior Coaching tibbi və ya psixoloji təcili yardımı "
            "əvəz etmir. Müraciətinizi məsul əməkdaşa yönləndirirəm."
        )

    # =====================================================
    # 4. Human agent
    # =====================================================

    if data.get(
        "intent"
    ) == "human_agent_request":

        lead[
            "status"
        ] = "ESCALATED"

        return (
            "Əlbəttə. Müraciətinizi məsul əməkdaşa "
            "yönləndirmək üçün qeydə aldım."
        )

    # =====================================================
    # 5. Real service complaint only
    # =====================================================

    if data.get(
        "intent"
    ) == "complaint":

        # main_concern mərhələsində qısa uşaq təsviri
        # complaint sayılmamalıdır.

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
    # 6. Greeting only
    # =====================================================

    if (
        data.get(
            "intent"
        ) == "greeting"
        and is_greeting(
            user_text
        )
    ):

        question = get_next_question(
            lead
        )

        return (
            "Salam 😊"
            + (
                "\n\n" + question
                if question
                else ""
            )
        )

    # =====================================================
    # 7. User asked a question
    #    ANSWER QUESTION FIRST
    # =====================================================

    if (
        data.get(
            "is_question"
        )
        or data.get(
            "intent"
        ) in [
            "faq_question",
            "program_interest",
            "meta_question",
        ]
    ):

        return answer_user_question(
            user_text=user_text,
            lead=lead,
            faq_min_score=faq_min_score,
        )

    # =====================================================
    # 8. If current field still unchanged and extractor
    #    found nothing useful, use fallback.
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
    # 9. Check completion
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
    # 10. MAXIMUM ONE QUESTION
    # =====================================================

    return get_next_question(
        lead
    )


# =========================================================
# 19. DATABASE
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

        # ---------------------------------------------
        # Leads migration
        # ---------------------------------------------

        existing = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(leads)"
            ).fetchall()
        }

        required = {

            "parent_title":
                "TEXT",

            "needs_concern_followup":
                "INTEGER DEFAULT 0",

            "concern_duration":
                "TEXT",

            "concern_onset":
                "TEXT",

            "children_json":
                "TEXT",
        }

        for column, dtype in required.items():

            if column not in existing:

                conn.execute(
                    f"""
                    ALTER TABLE leads
                    ADD COLUMN {column} {dtype}
                    """
                )

        # ---------------------------------------------
        # Logs migration
        # ---------------------------------------------

        existing_logs = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(conversation_logs)"
            ).fetchall()
        }

        log_required = {

            "intent":
                "TEXT",

            "confidence":
                "REAL",

            "faq_score":
                "REAL",

            "parent_title":
                "TEXT",

            "child_age":
                "INTEGER",

            "main_concern":
                "TEXT",

            "children_json":
                "TEXT",

            "preferred_call_time":
                "TEXT",

            "source":
                "TEXT",
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
# 20. TIME
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
# 21. FIND LEAD
# =========================================================

def find_lead_by_phone(
    phone: str,
):

    if not phone:

        return None

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

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

            return dict(
                row
            )

    return None


# =========================================================
# 22. SAVE LEAD
# =========================================================

def save_lead_to_db(
    lead: dict,
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
            [],
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
                            False,
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
# 23. SAVE CONVERSATION LOG
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
            [],
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
# 24. ADMIN HELPERS
# =========================================================

def get_all_leads():

    with sqlite3.connect(
        DB_PATH
    ) as conn:

        conn.row_factory = (
            sqlite3.Row
        )

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

        conn.row_factory = (
            sqlite3.Row
        )

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