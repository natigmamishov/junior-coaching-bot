"""
Junior Coaching — reqressiya testləri

İki hissədən ibarətdir:

1. OFFLINE testlər — OPENAI_API_KEY tələb etmir.
   Ad validasiyası, rol tanınması, telefon, FAQ recall.

2. LIVE ssenarilər — real LLM çağırışı edir.
   Müştəri rəyində qeyd olunan konkret hallar.

İşə salmaq:

    python3 test_scenarios.py           # yalnız offline
    python3 test_scenarios.py --live    # offline + LLM ssenariləri
"""

import sys

import bot_engine as bot
from conversation_core import OutputValidator, TurnOrchestrator
from core_contracts import PhoneStatus, TurnInput


PASSED = []
FAILED = []


def check(
    name: str,
    condition: bool,
    detail: str = "",
):

    if condition:

        PASSED.append(
            name
        )

        print(
            f"  ✓ {name}"
        )

    else:

        FAILED.append(
            (
                name,
                detail,
            )
        )

        print(
            f"  ✗ {name}"
            + (
                f"\n      {detail}"
                if detail
                else ""
            )
        )


# =========================================================
# OFFLINE — AD VALİDASİYASI
# =========================================================

def test_name_validation():

    print(
        "\nAd validasiyası (rəy: 'ad və rollar qarışır')"
    )

    # Ad OLMAYAN ifadələr rədd edilməlidir
    for value in [
        "Dedim yuxarıda",
        "anasıyam",
        "atasıyam",
        "mənə",
        "bilmirəm",
        "hə",
        "valideyniyəm",
    ]:

        check(
            f"'{value}' ad kimi qəbul edilmir",
            bot.clean_name(
                value
            ) is None,
            f"clean_name -> {bot.clean_name(value)!r}",
        )

    # Real adlar qorunmalıdır
    for value, expected in [
        ("Orxan", "Orxan"),
        ("Ayxan Məmmədov", "Ayxan Məmmədov"),
        ("Toğruldur", "Toğrul"),
        ("İsmayıldır", "İsmayıl"),
        ("Bahadır", "Bahadır"),
        ("Nadir", "Nadir"),
    ]:

        actual = bot.clean_name(
            value
        )

        check(
            f"'{value}' -> '{expected}'",
            actual == expected,
            f"alındı: {actual!r}",
        )


# =========================================================
# OFFLINE — ROL -> BAŞLIQ
# =========================================================

def test_parent_role():

    print(
        "\nRol tanınması"
    )

    for text, expected in [
        ("Salam, anasıyam Afət", "xanım"),
        ("atasıyam", "bəy"),
        ("nənəsiyəm", "xanım"),
        ("adım Leyladır", ""),
    ]:

        actual = bot.detect_parent_title_from_role(
            text
        )

        check(
            f"'{text}' -> {expected!r}",
            actual == expected,
            f"alındı: {actual!r}",
        )


# =========================================================
# OFFLINE — TELEFON
# =========================================================

def test_phone():

    print(
        "\nTelefon normallaşdırma"
    )

    for text, expected in [
        ("0501234567", "0501234567"),
        ("+994 50 123 45 67", "994501234567"),
        ("050 123 45 67", "0501234567"),
        ("12345", None),
    ]:

        actual = bot.normalize_phone(
            text
        )

        check(
            f"'{text}' -> {expected!r}",
            actual == expected,
            f"alındı: {actual!r}",
        )


# =========================================================
# OFFLINE — FAQ RECALL
# =========================================================

def test_faq_recall():

    print(
        "\nFAQ recall (rəy: 'sualın mənasını səhv tutur')"
    )

    # Doğru cavab namizədlər arasında OLMALIDIR ki,
    # LLM onu seçə bilsin.

    for question, marker in [
        ("görüşlər harada keçirilir?", "Məkan"),
        ("telefon zəngi nə qədər davam edir?", "dəqiqə"),
        ("görüşlər onlayn olur?", "onlayn"),
        ("buraxılan görüşün əvəzi olurmu?", "buraxsa"),
    ]:

        candidates = bot.retrieve_faq_candidates(
            question
        )

        found = any(
            marker.lower() in item["question"].lower()
            for item in candidates
        )

        check(
            f"'{question}' namizədlərində '{marker}' var",
            found,
            "namizədlər: "
            + ", ".join(
                item["question"][:35]
                for item in candidates
            ),
        )


# =========================================================
# OFFLINE — DÜZƏLİŞ MEXANİZMİ
# =========================================================

def test_corrections():

    print(
        "\nDüzəliş mexanizmi (rəy: 'state-i yeniləyə bilmir')"
    )

    lead = bot.create_empty_lead(
        "TEST"
    )

    lead["parent_name"] = "Afət"
    lead["children"][0]["name"] = "Səhv Ad"

    changed = bot.apply_corrections(
        lead,
        [
            {
                "field": "child_name",
                "child_index": 0,
                "value": "Tunar",
            }
        ],
    )

    check(
        "uşağın adı düzəlir",
        lead["children"][0]["name"] == "Tunar",
        f"alındı: {lead['children'][0]['name']!r}",
    )

    check(
        "düzəliş qeydə alınır",
        "child_name" in changed,
        f"changed={changed}",
    )

    # Boş slotun ilk dəfə dolması düzəliş SAYILMAMALIDIR
    lead2 = bot.create_empty_lead(
        "TEST"
    )

    changed2 = bot.apply_corrections(
        lead2,
        [
            {
                "field": "child_name",
                "child_index": 0,
                "value": "Ayxan",
            }
        ],
    )

    check(
        "ilk dolma düzəliş sayılmır",
        changed2 == [],
        f"changed={changed2}",
    )


# =========================================================
# OFFLINE — ÇOX UŞAQ STRUKTURU
# =========================================================

def test_multiple_children():

    print(
        "\nÇox uşaq strukturu"
    )

    lead = bot.create_empty_lead(
        "TEST"
    )

    bot.merge_children(
        lead,
        [
            {
                "name": "Ayxan",
                "age": 0,
                "main_concern": "",
            },
            {
                "name": "Orxan",
                "age": 0,
                "main_concern": "",
            },
        ],
        "2 usaqdir ayxan ve orxan",
    )

    names = [
        c.get("name")
        for c in lead["children"]
    ]

    check(
        "iki uşaq da saxlanılır",
        names == [
            "Ayxan",
            "Orxan",
        ],
        f"alındı: {names}",
    )

    # -------------------------------------------------
    # Rəy: "hamısı" cavabından sonra ikinci uşaq da
    # eyni qayğını və uydurulmuş yaşı alırdı.
    # -------------------------------------------------

    lead = bot.create_empty_lead(
        "TEST"
    )

    bot.ensure_child_slot(
        lead,
        0,
    ).update(
        {
            "name": "Ayxan",
            "age": 13,
        }
    )

    bot.ensure_child_slot(
        lead,
        1,
    )[
        "name"
    ] = "Orxan"

    lead[
        "active_child_index"
    ] = 0

    bot.merge_children(
        lead,
        [
            {
                "name": "Ayxan",
                "age": 13,
                "main_concern": "özgüvən",
            },
            {
                "name": "Orxan",
                "age": 15,
                "main_concern": "özgüvən",
            },
        ],
        "hamisi",
    )

    second = lead["children"][1]

    check(
        "aktiv uşağın qayğısı ikinciyə sızmır",
        second.get(
            "main_concern"
        ) is None,
        f"alındı: {second.get('main_concern')}",
    )

    check(
        "deyilməmiş yaş ikinci uşağa yazılmır",
        second.get(
            "age"
        ) is None,
        f"alındı: {second.get('age')}",
    )

    check(
        "aktiv uşaq öz cavabını alır",
        lead["children"][0].get(
            "main_concern"
        ) == "özgüvən",
        f"alındı: {lead['children'][0].get('main_concern')}",
    )

    # -------------------------------------------------
    # Mesajda keçməyən uşaq slotu açılmır.
    # -------------------------------------------------

    lead = bot.create_empty_lead(
        "TEST"
    )

    bot.merge_children(
        lead,
        [
            {
                "name": "Ayxan",
                "age": 13,
            },
            {
                "name": "Kamran",
                "age": 9,
            },
        ],
        "13",
    )

    check(
        "uydurulmuş uşaq əlavə edilmir",
        len(
            lead["children"]
        ) == 1,
        f"alındı: {[c.get('name') for c in lead['children']]}",
    )


# =========================================================
# OFFLINE — SƏRBƏST MƏTN VALİDASİYASI
# =========================================================

def test_fallback_guards():

    print(
        "\nSərbəst mətn sahələrinin qorunması"
    )

    lead = bot.create_empty_lead(
        "TEST"
    )

    bot.save_current_field_fallback(
        lead,
        "main_concern",
        "bilmirəm",
    )

    check(
        "'bilmirəm' main_concern kimi yazılmır",
        not lead["children"][0].get(
            "main_concern"
        ),
        f"alındı: {lead['children'][0].get('main_concern')!r}",
    )

    bot.save_current_field_fallback(
        lead,
        "child_name",
        "Dedim yuxarıda",
    )

    check(
        "'Dedim yuxarıda' ad kimi yazılmır",
        not lead["children"][0].get(
            "name"
        ),
        f"alındı: {lead['children'][0].get('name')!r}",
    )


# =========================================================
# OFFLINE — İMTİNA
# =========================================================

def test_refusal():

    print(
        "\nİmtina (rəy: eyni sual sonsuz təkrarlanırdı)"
    )

    for text in [
        "men telefonla elaqe ucun uygun deyilem",
        "vermək istəmirəm",
        "nömrə yazmaq istəmirəm",
        "lazım deyil",
    ]:

        check(
            f"'{text}' imtina kimi tanınır",
            bot.is_refusal(
                text
            ),
        )

    check(
        "'0501234567' imtina deyil",
        not bot.is_refusal(
            "0501234567"
        ),
    )

    # İkinci imtinadan sonra sahə keçilməlidir
    lead = bot.create_empty_lead(
        "TEST"
    )

    bot.handle_refusal(
        lead,
        "phone",
    )

    check(
        "birinci telefon imtinası dərhal qəbul edilir",
        "phone" in lead["_skipped_fields"] and lead["phone_declined"],
    )

    bot.handle_refusal(
        lead,
        "phone",
    )

    check(
        "ikinci imtinada telefon keçilir",
        "phone" in lead["_skipped_fields"],
        f"skipped={lead['_skipped_fields']}",
    )

    check(
        "telefon keçiləndə zəng vaxtı da keçilir",
        "preferred_call_time" in lead["_skipped_fields"],
        f"skipped={lead['_skipped_fields']}",
    )

    # İmtina mətni sahəyə yazılmamalıdır
    lead2 = bot.create_empty_lead(
        "TEST"
    )

    bot.save_current_field_fallback(
        lead2,
        "main_concern",
        "men telefonla elaqe ucun uygun deyilem",
    )

    check(
        "imtina mətni main_concern-ə yazılmır",
        not lead2["children"][0].get(
            "main_concern"
        ),
        f"alındı: {lead2['children'][0].get('main_concern')!r}",
    )


# =========================================================
# OFFLINE — "HAMISI"
# =========================================================

def test_concern_answer():

    print(
        "\n'hamısı' cavabı (FAQ sualı deyil)"
    )

    for text in [
        "hamisi",
        "hamısı",
        "hər biri",
        "özgüvən",
    ]:

        check(
            f"'{text}' birbaşa cavab kimi tanınır",
            bot.is_direct_concern_answer(
                text
            ),
        )

    check(
        "'qiymət nə qədərdir?' cavab deyil",
        not bot.is_direct_concern_answer(
            "qiymət nə qədərdir?"
        ),
    )


# =========================================================
# OFFLINE — TELEFONSUZ YEKUN
# =========================================================

def test_no_contact_finalization():

    print(
        "\nTelefonsuz yekunlaşma"
    )

    lead = bot.create_empty_lead(
        "TEST"
    )

    lead["parent_name"] = "Leyla"

    lead["children"][0].update(
        name="Sara",
        age=11,
        main_concern="özgüvən",
    )

    lead["_skipped_fields"] = [
        "phone",
        "preferred_call_time",
    ]

    check(
        "telefon keçiləndə axın bitir",
        bot.get_next_missing_field(
            lead
        ) is None,
        f"next={bot.get_next_missing_field(lead)!r}",
    )

    message = bot.finalize_lead(
        lead
    )

    check(
        "status NO_CONTACT olur",
        lead["status"] == "NO_CONTACT",
        f"status={lead['status']!r}",
    )

    check(
        "'zəng edəcəyik' vədi verilmir",
        "zəng" not in message.lower()
        or "əlaqə saxlaya bilməyəcəyik" in message,
        message[:110],
    )


# =========================================================
# LIVE — MÜŞTƏRİ RƏYİNDƏKİ SSENARİLƏR
# =========================================================

def run_dialog(
    messages,
):

    lead = bot.create_empty_lead(
        "TEST"
    )

    replies = []

    for message in messages:

        replies.append(
            bot.lead_agent_reply(
                message,
                lead,
            )
        )

    return (
        lead,
        replies,
    )


def test_live_scenarios():

    print(
        "\nLIVE ssenarilər (LLM)"
    )

    if bot.client is None:

        print(
            "  ! OPENAI_API_KEY yoxdur — keçilir"
        )

        return

    # 1. Bir mesajda bütün məlumatlar
    lead, _ = run_dialog(
        [
            "Mən Nərgizəm, oğlum Orxanın 15 yaşı var, "
            "özgüvəni zəifdir, 0501234567, "
            "sabah 15:00-dan sonra",
        ]
    )

    child = lead["children"][0]

    check(
        "bir mesajdan bütün slotlar çıxarılır",
        (
            lead["parent_name"] == "Nərgiz"
            and child.get("name") == "Orxan"
            and child.get("age") == 15
            and lead["phone"] == "0501234567"
            and bool(lead["preferred_call_time"])
        ),
        f"parent={lead['parent_name']!r} child={child} "
        f"phone={lead['phone']!r} time={lead['preferred_call_time']!r}",
    )

    # 2. Rol ad kimi götürülmür + düzəliş
    lead, _ = run_dialog(
        [
            "Salam, anasıyam",
            "Aygün mənəm, uşağın adı Ayxandır",
        ]
    )

    check(
        "'anasıyam' ad deyil, düzəliş tətbiq olunur",
        (
            lead["parent_name"] == "Aygün"
            and lead["children"][0].get("name") == "Ayxan"
        ),
        f"parent={lead['parent_name']!r} "
        f"child={lead['children'][0].get('name')!r}",
    )

    # 3. Sualın mənası
    lead, replies = run_dialog(
        [
            "görüşlər harada keçirilir?",
        ]
    )

    check(
        "'harada' sualına məkan cavabı verilir",
        any(
            word in replies[0]
            for word in [
                "ADAS",
                "Nərimanov",
                "küçəsi",
            ]
        ),
        replies[0][:110],
    )

    # 4. Danışıq ifadəsi KB-də axtarılmır
    _, replies = run_dialog(
        [
            "bir sual verə bilərəm?",
        ]
    )

    check(
        "'bir sual verə bilərəm?' -> buyurun",
        "buyurun" in replies[0].lower(),
        replies[0][:110],
    )

    # 5. İki uşaq
    lead, _ = run_dialog(
        [
            "Salam, adım Muraddır",
            "2 uşaqdır ayxan və orxan",
        ]
    )

    names = [
        c.get("name")
        for c in lead["children"]
        if c.get("name")
    ]

    check(
        "iki uşağın adı da saxlanılır",
        len(names) == 2,
        f"alındı: {names}",
    )

    # 6. Ekran 4-dəki döngü
    lead, replies = run_dialog(
        [
            "Salam, anasıyam afət",
            "Toğruldur birdəki sizdə görüşlər canlı olur yoxsa online",
            "Dedim yuxarıda",
        ]
    )

    check(
        "'Dedim yuxarıda' uşaq adı olmur",
        lead["children"][0].get(
            "name"
        ) == "Toğrul",
        f"alındı: {lead['children'][0].get('name')!r}",
    )

    # 7. Telefon imtinası döngə yaratmır
    lead, replies = run_dialog(
        [
            "salam adim leyladir",
            "qizim sara",
            "11",
            "hamisi",
            "men telefonla elaqe ucun uygun deyilem",
            "vermek istemirem",
        ]
    )

    check(
        "telefon imtinası döngə yaratmır",
        lead["status"] == "NO_CONTACT",
        f"status={lead['status']!r} skipped={lead['_skipped_fields']}",
    )

    check(
        "imtina ESCALATED etmir",
        lead["status"] != "ESCALATED",
        f"status={lead['status']!r}",
    )

    check(
        "'hamısı' FAQ cavabı qaytarmır",
        "Akademik" not in replies[3],
        replies[3][:110],
    )


# =========================================================
# MAIN
# =========================================================

def test_core_engine_contract():
    print("\nCore engine contract")

    lead = bot.create_empty_lead("TEST")
    for key in (
        "lead_stage", "application_status", "objections", "questions",
        "previous_actions", "pending_actions", "handoff_status", "owner",
    ):
        check(f"state contains {key}", key in lead, repr(lead))

    analysis = bot.verify_analysis({
        "intent": "field_answer",
        "confidence": 0.2,
        "parent_name": "Bu Nomreyle",
        "children": [],
        "preferred_call_time": "",
    })
    check(
        "low-confidence entity requests clarification",
        analysis["clarification_needed"] and not analysis["parent_name"],
        repr(analysis),
    )

    lead.update({
        "parent_name": "Aynur", "phone": "0501234567",
        "preferred_call_time": "sabah 15:00",
    })
    lead["children"] = [{
        "name": "Murad", "age": 15, "main_concern": "məktəbdə danışmır",
        "needs_concern_followup": False, "concern_duration": None,
        "concern_onset": None,
    }]
    bot.update_conversation_state(lead, {
        "questions": ["Görüşlər haradadır?", "Hansı günlərdir?"],
        "objections": ["Qiymət tərəddüdü"],
        "handoff_required": False,
    })
    check("multi-intent questions persist", len(lead["questions"]) == 2, repr(lead))
    check("objection persists", lead["objections"] == ["Qiymət tərəddüdü"], repr(lead))
    check(
        "completed lead exposes callback action",
        lead["application_status"] == "completed"
        and "create_callback" in lead["pending_actions"],
        repr(lead),
    )

    bot.update_conversation_state(lead, {"handoff_required": True})
    check(
        "handoff changes ownership",
        lead["owner"] == "human" and "human_handoff" in lead["pending_actions"],
        repr(lead),
    )


def test_customer_feedback_round_two():
    print("\nCustomer feedback round 2")
    lead = bot.create_empty_lead("TEST")
    lead["phone"] = "0554445559"
    lead["children"][0]["age"] = 15

    text = (
        "Mən sizə hansı telefon nömrəsini və oğlumun neçə yaşında olduğunu "
        "demişdim? Bir də mənim adımı bilirsiniz?"
    )
    fields = bot._detect_requested_state_fields(text)
    answer = bot.answer_requested_state_fields(lead, fields)
    check(
        "multi-field recall detects phone, age and parent name",
        fields == ["phone", "child_age", "parent_name"],
        repr(fields),
    )
    check(
        "multi-field recall returns known and missing values together",
        "0554445559" in answer and "15" in answer and "qeyd edilməyib" in answer,
        answer,
    )

    payment = bot.answer_special_question("İlkin tanışlıq ödənişlidir?", lead)
    check(
        "unknown binary payment fact is not replaced by nearby FAQ",
        payment is not None and "dəqiq fakt yoxdur" in payment,
        repr(payment),
    )

    first = bot._callback_reference_reply(
        "Bu gün oğlum yanımda deyil, olar ki sabah edək?", [], lead
    )
    check(
        "child absence plus tomorrow resolves to callback scheduling",
        first is not None and lead["preferred_call_time"] == "sabah"
        and "Hansı saat" in first,
        repr(first),
    )

    lead["preferred_call_time"] = None
    followup = bot._callback_reference_reply(
        "zəngi nəzərdə tuturdum",
        [{"role": "user", "content": "Bu gün oğlum yanımda deyil, olar ki sabah edək?"}],
        lead,
    )
    check(
        "elliptical correction inherits tomorrow from previous turn",
        followup is not None and lead["preferred_call_time"] == "sabah",
        repr(followup),
    )

    duration = bot._callback_reference_reply("Zəng neçə dəqiqə davam edir?", [], lead)
    check(
        "call-duration question is not mistaken for rescheduling",
        duration is None,
        repr(duration),
    )

    greeting_analysis = bot.verify_analysis(
        {
            "intent": "greeting", "confidence": 0.1,
            "parent_name": "Salam",
            "children": [{"name": "", "age": 0, "main_concern": ""}],
            "preferred_call_time": "", "clarification_needed": True,
            "clarification_question": "Məlumat kimə aiddir?",
        },
        user_text="Salam",
    )
    check(
        "pure greeting never triggers entity clarification",
        greeting_analysis["intent"] == "greeting"
        and not greeting_analysis["clarification_needed"]
        and not greeting_analysis["parent_name"]
        and greeting_analysis["children"] == [],
        repr(greeting_analysis),
    )

    sibling_answer = bot.answer_special_question(
        "Bir ailədən 2 uşaq gələ bilərmi?", lead
    )
    check(
        "two-child policy question gets specific safe answer",
        sibling_answer is not None
        and "dəqiq göstərilməyib" in sibling_answer
        and "Hər iki uşağın yaşını" in sibling_answer,
        repr(sibling_answer),
    )

    sibling_history = [
        {"role": "user", "content": "Bir ailədən 2 uşaq gələ bilərmi?"},
        {"role": "assistant", "content": sibling_answer or ""},
    ]
    ages = bot._contextual_bare_child_ages("14 16", sibling_history)
    check(
        "bare ages resolve to two children from conversation context",
        ages == [14, 16],
        repr(ages),
    )
    check(
        "bare ages without child context are not guessed",
        bot._contextual_bare_child_ages("14 16", []) == [],
        "unexpected extraction",
    )

    sibling_lead = bot.create_empty_lead("TEST")
    bot.merge_extracted_information(
        sibling_lead,
        {
            "children": [
                {"name": "", "age": age, "main_concern": ""}
                for age in ages
            ],
            "multiple_children": True,
            "children_count": len(ages),
            "corrections": [], "parent_name": "", "parent_title": "",
            "phone": "", "preferred_call_time": "",
        },
        "14 16",
    )
    check(
        "both contextual ages persist in separate child state",
        [child.get("age") for child in sibling_lead["children"]] == [14, 16],
        repr(sibling_lead["children"]),
    )

    clarification_history = [{
        "role": "assistant",
        "content": "Düzgün qeyd etməyim üçün məlumatın kimə aid olduğunu dəqiqləşdirə bilərsiniz?",
    }]
    clarification_reply = bot._clarification_reference_reply(
        "Hansı məlumat?", clarification_history
    )
    check(
        "agent resolves reference to its own clarification",
        clarification_reply is not None
        and "Əvvəlki mesajınızda" in clarification_reply
        and "buna ehtiyac yoxdur" in clarification_reply,
        repr(clarification_reply),
    )

    price_reply = bot.answer_special_question("Qiymət nə qədərdir?", lead)
    check(
        "generic price answer does not re-ask which program",
        price_reply is not None
        and "vahid məbləğ" in price_reply
        and "hansı proqram" not in price_reply.lower(),
        repr(price_reply),
    )

    original_analyze = bot.analyze_message
    try:
        bot.analyze_message = lambda user_text, lead, history=None, faq_candidates=None: (
            bot.build_fallback_extraction(user_text)
        )
        full_lead = bot.create_empty_lead("TEST")
        full_history = [
            {"role": "user", "content": "Bir ailədən 2 uşaq gələ bilərmi?"},
            {"role": "assistant", "content": sibling_answer or ""},
        ]
        full_reply = bot.lead_agent_reply("14 16", full_lead, history=full_history)
    finally:
        bot.analyze_message = original_analyze

    check(
        "full engine returns a response after contextual bare ages",
        bool(full_reply.strip()),
        repr(full_reply),
    )
    check(
        "full engine stores both ages without crashing",
        [child.get("age") for child in full_lead["children"]] == [14, 16],
        repr(full_lead["children"]),
    )

    compound = bot._expand_compound_faq_questions(
        "Proqram nə qədər vaxt çəkir harada keçirilir hansı saatdadır və s.",
        ["Proqram nə qədər vaxt çəkir harada keçirilir hansı saatdadır və s."],
    )
    check(
        "punctuation-free multi-intent expands all FAQ topics",
        len(compound) == 3
        and any("davam" in q for q in compound)
        and any("harada" in q for q in compound)
        and any("saat" in q for q in compound),
        repr(compound),
    )

    nine_month_price = bot.answer_special_question(
        "Qiyməti nə qədərdir 9 aylıq proqramın?", lead
    )
    check(
        "nine-month generic price question gets direct safe answer",
        nine_month_price is not None
        and "vahid məbləğ" in nine_month_price
        and "əsas ehtiyac" not in nine_month_price,
        repr(nine_month_price),
    )

    no_phone_lead = bot.create_empty_lead("TEST")
    no_phone_lead["parent_name"] = "Günel"
    no_phone_lead["children"][0].update({
        "name": "Murad", "age": 15, "main_concern": "özgüvən",
    })
    original_analyze = bot.analyze_message
    try:
        bot.analyze_message = lambda user_text, lead, history=None, faq_candidates=None: {
            **bot.build_fallback_extraction(user_text),
            "intent": "refusal", "confidence": 1.0,
        }
        no_phone_reply = bot.lead_agent_reply(
            "Xeyr, nömrəsiz davam edək", no_phone_lead, history=[]
        )
    finally:
        bot.analyze_message = original_analyze

    check(
        "explicit no-phone preference is accepted on first turn",
        no_phone_lead["status"] == "NEW"
        and no_phone_lead["phone_declined"] is True
        and "phone" in no_phone_lead["_skipped_fields"],
        repr(no_phone_lead),
    )
    check(
        "no-contact response is not duplicated by finalization",
        "nömrənizi" not in no_phone_reply.lower()
        and "məlumatlarınızı qeyd etdim" not in no_phone_reply.lower()
        and "buradan davam" in no_phone_reply.lower(),
        repr(no_phone_reply),
    )

    check(
        "phone refusal is persisted as an explicit preference",
        no_phone_lead["phone_declined"] is True
        and bot.get_next_missing_field(no_phone_lead) is None,
        repr(no_phone_lead),
    )

    bot.merge_extracted_information(
        no_phone_lead,
        {
            "corrections": [], "parent_name": "", "parent_title": "",
            "children": [], "multiple_children": False, "children_count": 0,
            "phone": "0501234567", "preferred_call_time": "",
        },
        "Fikrimi dəyişdim, nömrəm 0501234567",
    )
    check(
        "voluntarily supplied phone reverses declined preference",
        no_phone_lead["phone"] == "0501234567"
        and no_phone_lead["phone_declined"] is False,
        repr(no_phone_lead),
    )
    check(
        "reversed phone preference reopens callback flow",
        no_phone_lead["status"] == "NEW"
        and bot.get_next_missing_field(no_phone_lead) == "preferred_call_time",
        repr(no_phone_lead),
    )

    refinement_lead = bot.create_empty_lead("TEST")
    refinement_lead["children"][0]["main_concern"] = "ünsiyyət"
    refined = bot.merge_extracted_information(
        refinement_lead,
        {
            "intent": "correction",
            "corrections": [{
                "field": "main_concern", "child_index": 0, "value": "özgüvən",
            }],
            "parent_name": "", "parent_title": "", "children": [],
            "multiple_children": False, "children_count": 0,
            "phone": "", "preferred_call_time": "",
        },
        "özgüvən",
    )
    check(
        "new concern refinement is not acknowledged as correction",
        refinement_lead["children"][0]["main_concern"] == "özgüvən"
        and refined == [],
        repr((refined, refinement_lead["children"])),
    )

    composition_lead = bot.create_empty_lead("TEST")
    original_generate = bot.generate_contextual_kb_answer
    try:
        bot.generate_contextual_kb_answer = lambda question, **kwargs: {
            "Proqram nə qədər davam edir?": "Proqram 9 ay davam edir.",
            "Görüşlər harada keçirilir?": "Görüşlər ADAS Plaza-da keçirilir.",
            "Görüşlərin gün və saatları necə müəyyən edilir?": (
                "Saatla bağlı bazamda dəqiq məlumat yoxdur."
            ),
        }.get(question, "Məlumat yoxdur.")
        composed = bot.answer_user_question(
            user_text="Proqram nə qədər çəkir, harada keçirilir, hansı saatdadır?",
            lead=composition_lead,
            faq_min_score=0.2,
            data={"questions": [
                "Proqram nə qədər davam edir?", "Görüşlər harada keçirilir?",
                "Görüşlərin gün və saatları necə müəyyən edilir?",
            ]},
        )
    finally:
        bot.generate_contextual_kb_answer = original_generate
    check(
        "multi-intent answers are composed as one message",
        "\n\n" not in composed
        and "9 ay" in composed and "ADAS Plaza" in composed and "Saatla" in composed,
        repr(composed),
    )
    check(
        "answered questions leave pending state",
        composition_lead["pending_questions"] == []
        and len(composition_lead["resolved_questions"]) == 3,
        repr(composition_lead),
    )

    chat_lead = bot.create_empty_lead("TEST")
    chat_lead["parent_name"] = "Günay"
    chat_lead["children"][0].update({"name": "Tahir", "age": 14})
    original_analyze = bot.analyze_message
    try:
        bot.analyze_message = lambda user_text, lead, history=None, faq_candidates=None: {
            **bot.build_fallback_extraction(user_text),
            "intent": "field_answer", "confidence": 1.0,
        }
        chat_reply = bot.lead_agent_reply(
            "Xeyr, buradan cavablayın", chat_lead, history=[]
        )
    finally:
        bot.analyze_message = original_analyze
    check(
        "chat-only request permanently disables phone funnel",
        chat_lead["phone_declined"] is True
        and chat_lead["contact_requested"] is False
        and "telefon" not in chat_reply.lower()
        and "nömr" not in chat_reply.lower(),
        repr((chat_reply, chat_lead)),
    )
    check(
        "chat-only discovery is not marked completed",
        chat_lead["status"] == "NEW"
        and chat_lead["application_status"] == "in_progress",
        repr(chat_lead),
    )

    recall_fields = bot._detect_requested_state_fields(
        "Yaşını yuxarıda qeyd etmişəm, narahatlığımı da"
    )
    check(
        "age and concern recall request keeps both fields",
        recall_fields == ["child_age", "main_concern"],
        repr(recall_fields),
    )

def test_production_architecture_contract():
    print("\nProduction architecture contract")
    state = bot.create_empty_lead("TEST")
    calls = {"count": 0}

    def handler(user_text, lead, faq_min_score, history):
        calls["count"] += 1
        lead["parent_name"] = "Aynur"
        return "Salam, Aynur xanım."

    orchestrator = TurnOrchestrator()
    turn = TurnInput(
        conversation_id="conversation-1",
        channel="test",
        channel_message_id="message-1",
        text="Mən Aynuram",
    )
    first = orchestrator.process(turn, state, [], handler, 0.2)
    second = orchestrator.process(turn, state, [], handler, 0.2)

    check("canonical schema version exists", state["schema_version"] == 1, repr(state))
    check("state version increments once", state["state_version"] == 1, repr(state))
    check("duplicate message does not run handler", calls["count"] == 1, repr(calls))
    check("duplicate returns original response", second.duplicate and second.response == first.response, repr(second))
    check("turn trace stores state diff", bool(state["_turn_traces"][0]["state_diff"]), repr(state["_turn_traces"]))
    check("turn trace stores build versions", state["_last_turn_trace"]["kb_version"] == "faq-v1", repr(state["_last_turn_trace"]))
    check("children receive stable IDs", bool(state["children"][0]["child_id"]), repr(state["children"]))
    check("phone status is canonical enum value", state["phone_status"] == PhoneStatus.UNKNOWN.value, repr(state))

    validation = OutputValidator().validate("", [])
    check("output validator blocks empty response", not validation["valid"] and "empty_response" in validation["violations"], repr(validation))


def test_consultative_discovery_order():
    print("\nConsultative discovery order")

    lead = bot.create_empty_lead("TEST")
    check(
        "first discovery question is the parent's concern",
        bot.get_next_missing_field(lead) == "main_concern",
        repr(lead),
    )

    child = bot.get_active_child(lead)
    child["main_concern"] = "məktəbdə özünü ifadə etmək"
    bot.sync_flat_fields(lead)
    check(
        "age follows the concern",
        bot.get_next_missing_field(lead) == "child_age",
        repr(lead),
    )

    child["age"] = 14
    bot.sync_flat_fields(lead)
    check(
        "names and phone are not required during discovery",
        bot.get_next_missing_field(lead) is None,
        repr(lead),
    )

    lead["contact_requested"] = True
    check(
        "name collection opens only in explicit contact stage",
        bot.get_next_missing_field(lead) == "parent_name",
        repr(lead),
    )

    lead["parent_name"] = "Aysel"
    check(
        "child name is collected late with contact details",
        bot.get_next_missing_field(lead) == "child_name",
        repr(lead),
    )

    child["name"] = "Tural"
    bot.sync_flat_fields(lead)
    check(
        "phone follows late-stage names",
        bot.get_next_missing_field(lead) == "phone",
        repr(lead),
    )

    bot.continue_without_phone(lead)
    check(
        "unanswered phone prompt is not repeated",
        bot.get_next_missing_field(lead) is None and not lead["contact_requested"],
        repr(lead),
    )

    acknowledgement = bot.build_field_ack("main_concern", lead)
    check(
        "field answers receive a human acknowledgement",
        acknowledgement.startswith("Anladım"),
        acknowledgement,
    )

    check(
        "mere program interest does not open contact funnel",
        not bot.is_explicit_contact_request(
            "Salam, 15 yaşlı oğlum üçün maraqlanıram"
        ),
    )
    check(
        "explicit registration opens contact funnel",
        bot.is_explicit_contact_request("Qeydiyyatdan keçmək istəyirəm"),
    )

    overview = bot.answer_special_question(
        "Adım Aygündür. Proqram barədə qısa məlumat verə bilərsiniz?",
        lead,
    )
    check(
        "general program overview gets a direct approved answer",
        bool(overview) and "12–18" in overview and "inkişaf proqramıdır" in overview,
        repr(overview),
    )

    check(
        "explicit age correction is detected without LLM",
        bot.explicit_age_correction(
            "Yeri gəlmişkən, səhv demişəm, oğlum 15 yox, 16 yaşındadır."
        ) == 16,
    )

    check(
        "detailed information request stays in assistant",
        bot.is_program_overview_request("Ətraflı məlumat almaq istəyirəm"),
    )
    check(
        "presence check is recognized",
        bot.is_presence_check("burdasız?"),
    )

    handed_off = bot.create_empty_lead("TEST")
    handed_off["owner"] = "human"
    handed_off["handoff_status"] = "requested"
    check(
        "old handoff state does not freeze later conversation",
        bot.decide_next_step_policy(
            handed_off,
            {"intent": "field_answer", "handoff_required": False,
             "clarification_needed": False, "ready_to_proceed": False},
        ) == "CONTINUE",
    )

    bare_age_lead = bot.create_empty_lead("TEST")
    bare_age_lead["children"][0]["main_concern"] = "ünsiyyət"
    bare_age_reply = bot._process_legacy_turn("16", bare_age_lead, history=[])
    check(
        "bare age answer is saved and answered without LLM",
        bare_age_lead["children"][0]["age"] == 16 and bool(bare_age_reply),
        f"lead={bare_age_lead!r} reply={bare_age_reply!r}",
    )

    check(
        "contact method question is separated from callback scheduling",
        bot.is_contact_method_question("Sizinlə necə əlaqə saxlamaq olar?"),
    )
    contact_answer = bot.answer_special_question(
        "Buradan zəng edəcəksiniz, yoxsa nömrəyə?", bare_age_lead
    )
    check(
        "contact channel gets a direct answer",
        bool(contact_answer) and "telefon nömrəsi" in contact_answer,
        repr(contact_answer),
    )

    self_contact_lead = bot.create_empty_lead("TEST")
    self_contact_lead["contact_requested"] = True
    self_contact_reply = bot._process_legacy_turn(
        "Sonra özüm zəng edərəm", self_contact_lead, history=[]
    )
    check(
        "self-contact preference disables callback funnel",
        self_contact_lead["phone_declined"]
        and not self_contact_lead["contact_requested"]
        and "hansı tarixdə" not in self_contact_reply.lower(),
        f"lead={self_contact_lead!r} reply={self_contact_reply!r}",
    )

    clinical = bot.answer_special_question(
        "Oğlumda panik atak var, kömək edə bilərsiniz?", bare_age_lead
    )
    check(
        "panic attack never receives a treatment promise",
        clinical is not None
        and "terapiya" in clinical
        and "müalicə etdiyini iddia edə bilməz" in clinical,
        repr(clinical),
    )

    audience = bot.answer_special_question(
        "Siz ancaq uşaqlarla işləyirsiniz?", bare_age_lead
    )
    check(
        "audience question answers the approved age range",
        audience is not None and "12–18" in audience,
        repr(audience),
    )

    check(
        "chat preference accepts informal request",
        bot.prefers_chat_only("Zəhmət olmasa buradan yazın da"),
    )

    vague_price = bot.answer_special_question("qiymət vəss", bare_age_lead)
    check(
        "informal vague price request gets safe price answer",
        vague_price is not None and "vahid məbləğ" in vague_price,
        repr(vague_price),
    )


def main():

    print(
        "Junior Coaching — reqressiya testləri"
    )

    test_name_validation()
    test_parent_role()
    test_phone()
    test_faq_recall()
    test_corrections()
    test_multiple_children()
    test_fallback_guards()
    test_refusal()
    test_concern_answer()
    test_no_contact_finalization()
    test_core_engine_contract()
    test_customer_feedback_round_two()
    test_production_architecture_contract()
    test_consultative_discovery_order()

    if "--live" in sys.argv:

        test_live_scenarios()

    print(
        f"\n{'='*50}"
    )

    print(
        f"Keçdi: {len(PASSED)}   Uğursuz: {len(FAILED)}"
    )

    if FAILED:

        print(
            "\nUğursuz testlər:"
        )

        for name, detail in FAILED:

            print(
                f"  - {name}"
            )

        return 1

    return 0


if __name__ == "__main__":

    sys.exit(
        main()
    )
