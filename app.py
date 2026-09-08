import hmac
import os
import uuid
import sqlite3
import time

from pathlib import Path

import pandas as pd
import streamlit as st

import bot_engine as bot


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Junior Coaching",
    page_icon="💬",
    layout="centered",
)


# =========================================================
# CONSTANTS
# =========================================================

WELCOME_MESSAGE = (
    "Salam,\n\n"
    "Övladınıza uyğun proqramı öyrənmək üçün sizə 2 qısa sualımız var.\n\n"
    "İlk olaraq, yaşını qeyd edə bilərsiniz?"
)


# =========================================================
# SESSION INITIALIZATION
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(
        uuid.uuid4()
    )


if "lead" not in st.session_state:
    st.session_state.lead = (
        bot.create_empty_lead(
            source="Streamlit"
        )
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]


if "conversation_finished" not in st.session_state:
    st.session_state.conversation_finished = False


if "lead_saved" not in st.session_state:
    st.session_state.lead_saved = False

if "need_batch_active" not in st.session_state:
    st.session_state.need_batch_active = False

if "need_batch_deadline" not in st.session_state:
    st.session_state.need_batch_deadline = 0.0


# =========================================================
# HELPER — ADMIN PASSWORD
# =========================================================

def secrets_file_exists() -> bool:

    """
    Streamlit-in secrets.toml axtardığı yerlər.

    Streamlit Cloud da şifrələri ev qovluğundakı
    bu fayla yazır, ona görə yoxlama hər iki mühitdə işləyir.
    """

    candidates = [
        Path.home()
        / ".streamlit"
        / "secrets.toml",

        Path(__file__).parent
        / ".streamlit"
        / "secrets.toml",

        Path.cwd()
        / ".streamlit"
        / "secrets.toml",
    ]

    return any(
        path.is_file()
        for path in candidates
    )


def get_admin_password():

    """
    Şifrəni əvvəlcə Streamlit Secrets-dən,
    sonra mühit dəyişənindən oxuyur.

    Heç biri təyin edilməyibsə None qaytarır.

    QEYD: fayl yoxdursa st.secrets-ə ÜMUMİYYƏTLƏ
    toxunmuruq. Streamlit "No secrets files found"
    xətasını exception atmazdan ƏVVƏL ekrana yazır,
    ona görə try/except onu gizlədə bilmir —
    ADMIN_PASSWORD mühit dəyişəni ilə düzgün işləyən
    quraşdırmada da qırmızı xəta görünürdü.
    """

    if secrets_file_exists():

        try:

            secret = st.secrets.get(
                "ADMIN_PASSWORD"
            )

            if secret:
                return str(
                    secret
                )

        except Exception:

            # Fayl var, amma oxunmur/sınıqdır.
            pass

    return os.environ.get(
        "ADMIN_PASSWORD"
    )


# =========================================================
# HELPER — PERSIST LEAD
# =========================================================

def persist_lead_once(
    lead: dict,
):

    """
    Leadi bir dəfə bazaya yazır.

    CALL_REQUESTED üçün istifadəçiyə müraciət nömrəsi qaytarır.
    ESCALATED üçün lead yenə saxlanılır, lakin əlavə
    təsdiq mesajı göstərilmir (None qaytarır).
    """

    if st.session_state.lead_saved:
        # The chat stays open after completion, so later corrections must also
        # be reflected in persistent storage.
        try:
            bot.update_lead_in_db(lead)
        except Exception as exc:
            print("LEAD UPDATE ERROR:", exc)
        return None

    status = lead.get(
        "status"
    )

    phone = lead.get(
        "phone"
    )

    existing_lead = (
        bot.find_lead_by_phone(
            phone
        )
        if phone
        else None
    )

    if existing_lead is not None:

        st.session_state.lead_saved = True
        lead["_db_id"] = existing_lead["id"]
        bot.update_lead_in_db(lead)

        if status == "CALL_REQUESTED":

            return (
                "Bu telefon nömrəsi ilə artıq "
                f"{existing_lead['id']} nömrəli "
                "müraciət mövcuddur. "
                "Məlumatınız komanda tərəfindən "
                "nəzərə alınacaq."
            )

        return None

    try:

        lead_id = bot.save_lead_to_db(
            lead
        )

        st.session_state.lead_saved = True

        if status == "CALL_REQUESTED":

            return (
                "Müraciətiniz sistemdə qeyd edildi. "
                f"Müraciət nömrəniz: {lead_id}"
            )

        return None

    except Exception as exc:

        print(
            "LEAD SAVE ERROR:",
            exc,
        )

        if status == "CALL_REQUESTED":

            return (
                "Müraciət məlumatlarınız toplandı, "
                "lakin sistemdə saxlanarkən "
                "texniki problem yarandı."
            )

        return None


# =========================================================
# HELPER — RESET CHAT
# =========================================================

def reset_chat():

    st.session_state.session_id = str(
        uuid.uuid4()
    )

    st.session_state.lead = (
        bot.create_empty_lead(
            source="Streamlit"
        )
    )

    st.session_state.messages = [
        {
            "role": "assistant",
            "content": WELCOME_MESSAGE,
        }
    ]

    st.session_state.conversation_finished = False

    st.session_state.lead_saved = False
    st.session_state.need_batch_active = False
    st.session_state.need_batch_deadline = 0.0


# =========================================================
# ADMIN SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Junior Coaching")

    st.caption(
        "Test və admin panel"
    )

    st.divider()

    st.subheader("Admin")

    admin_password = st.text_input(
        "Admin şifrəsi",
        type="password",
        key="admin_password",
    )

    # Şifrə koda yazılmır.
    # .streamlit/secrets.toml faylından
    # və ya ADMIN_PASSWORD mühit dəyişənindən oxunur.
    ADMIN_PASSWORD = get_admin_password()

    if admin_password:

        if not ADMIN_PASSWORD:

            st.error(
                "Admin şifrəsi konfiqurasiya edilməyib. "
                "`.streamlit/secrets.toml` faylında "
                "`ADMIN_PASSWORD` təyin edin "
                "(nümunə: `.streamlit/secrets.toml.example`)."
            )

        elif hmac.compare_digest(
            admin_password,
            ADMIN_PASSWORD,
        ):

            st.success(
                "Admin giriş aktivdir"
            )

            try:

                with sqlite3.connect(
                    bot.DB_PATH
                ) as conn:

                    leads_df = pd.read_sql_query(
                        """
                        SELECT *
                        FROM leads
                        ORDER BY id DESC
                        """,
                        conn,
                    )

                    # Hər uşaq ayrıca sətir.
                    # leads cədvəlindəki child_name yalnız
                    # BİRİNCİ uşağı göstərir, ona görə
                    # ikinci uşaq burada axtarılmalıdır.
                    children_df = pd.read_sql_query(
                        """
                        SELECT
                            c.lead_id,
                            l.parent_name,
                            l.parent_title,
                            l.phone,
                            l.status,
                            c.child_index,
                            c.name,
                            c.age,
                            c.main_concern,
                            c.needs_concern_followup,
                            c.concern_duration,
                            c.concern_onset,
                            c.created_at
                        FROM children AS c
                        JOIN leads AS l
                            ON l.id = c.lead_id
                        ORDER BY
                            c.lead_id DESC,
                            c.child_index
                        """,
                        conn,
                    )

                    logs_df = pd.read_sql_query(
                        """
                        SELECT *
                        FROM conversation_logs
                        ORDER BY id DESC
                        """,
                        conn,
                    )


                # -----------------------------------------
                # SUMMARY
                # -----------------------------------------

                st.metric(
                    "Lead sayı",
                    len(leads_df),
                )

                st.metric(
                    "Uşaq sayı",
                    len(children_df),
                )

                st.metric(
                    "Log sayı",
                    len(logs_df),
                )


                if (
                    not logs_df.empty
                    and "session_id"
                    in logs_df.columns
                ):

                    session_count = (
                        logs_df[
                            "session_id"
                        ]
                        .nunique()
                    )

                    st.metric(
                        "Test sessiyası",
                        session_count,
                    )


                st.divider()


                # -----------------------------------------
                # LEADS
                # -----------------------------------------

                st.subheader(
                    "Leads"
                )

                if leads_df.empty:

                    st.info(
                        "Hələ lead yoxdur."
                    )

                else:

                    st.dataframe(
                        leads_df,
                        use_container_width=True,
                        hide_index=True,
                    )


                    leads_csv = (
                        leads_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8-sig"
                        )
                    )


                    st.download_button(
                        label="Leads CSV yüklə",
                        data=leads_csv,
                        file_name=(
                            "junior_coaching_leads.csv"
                        ),
                        mime="text/csv",
                    )


                st.divider()


                # -----------------------------------------
                # CHILDREN
                # -----------------------------------------

                st.subheader(
                    "Uşaqlar"
                )

                st.caption(
                    "Leads cədvəlindəki `child_name` yalnız "
                    "birinci uşağı göstərir. "
                    "Bütün uşaqlar bu siyahıdadır."
                )

                if children_df.empty:

                    st.info(
                        "Hələ uşaq qeydi yoxdur."
                    )

                else:

                    st.dataframe(
                        children_df,
                        use_container_width=True,
                        hide_index=True,
                    )


                    children_csv = (
                        children_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8-sig"
                        )
                    )


                    st.download_button(
                        label="Uşaqlar CSV yüklə",
                        data=children_csv,
                        file_name=(
                            "junior_coaching_children.csv"
                        ),
                        mime="text/csv",
                    )


                st.divider()


                # -----------------------------------------
                # CONVERSATION LOGS
                # -----------------------------------------

                st.subheader(
                    "Conversation Logs"
                )


                if logs_df.empty:

                    st.info(
                        "Hələ conversation log yoxdur."
                    )

                else:

                    st.dataframe(
                        logs_df,
                        use_container_width=True,
                        hide_index=True,
                    )


                    logs_csv = (
                        logs_df
                        .to_csv(
                            index=False
                        )
                        .encode(
                            "utf-8-sig"
                        )
                    )


                    st.download_button(
                        label=(
                            "Conversation Logs CSV yüklə"
                        ),
                        data=logs_csv,
                        file_name=(
                            "junior_coaching_conversation_logs.csv"
                        ),
                        mime="text/csv",
                    )


                st.divider()


                # -----------------------------------------
                # SQLITE DB DOWNLOAD
                # -----------------------------------------

                try:

                    with open(
                        bot.DB_PATH,
                        "rb",
                    ) as db_file:

                        db_bytes = (
                            db_file.read()
                        )


                    st.download_button(
                        label=(
                            "SQLite DB yüklə"
                        ),
                        data=db_bytes,
                        file_name=(
                            "junior_coaching.db"
                        ),
                        mime=(
                            "application/octet-stream"
                        ),
                    )

                except Exception as exc:

                    st.warning(
                        f"DB faylı yüklənə bilmədi: {exc}"
                    )


            except Exception as exc:

                st.error(
                    f"Admin məlumatları oxuna bilmədi: {exc}"
                )

        else:

            st.error(
                "Admin şifrəsi yanlışdır."
            )


# =========================================================
# MAIN UI
# =========================================================

st.title(
    "Junior Coaching"
)

st.caption(
    "Junior Coaching proqramı üzrə virtual müraciət köməkçisi"
)


# =========================================================
# CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# =========================================================
# NEED MESSAGE DEBOUNCE
# =========================================================

@st.fragment(run_every=1)
def flush_need_batch():
    """After eight seconds, ask for the parent name exactly once."""
    if not st.session_state.need_batch_active:
        return
    if time.time() < st.session_state.need_batch_deadline:
        return

    st.session_state.need_batch_active = False
    if (
        st.session_state.lead.get("status") != "STOPPED"
        and bot.get_next_missing_field(st.session_state.lead) == "parent_name"
    ):
        prompt = bot.get_next_question(st.session_state.lead)
        last = st.session_state.messages[-1] if st.session_state.messages else {}
        if prompt and not (
            last.get("role") == "assistant"
            and last.get("content") == prompt
        ):
            st.session_state.messages.append(
                {"role": "assistant", "content": prompt}
            )
    st.rerun()


flush_need_batch()


# =========================================================
# CHAT INPUT
# =========================================================

if True:  # A completed application does not close the conversation.

    user_text = st.chat_input(
        "Mesajınızı yazın..."
    )


    if user_text:

        user_text = (
            user_text.strip()
        )


        if user_text:

            # ---------------------------------------------
            # USER MESSAGE
            # ---------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": user_text,
                    "message_id": str(uuid.uuid4()),
                }
            )


            with st.chat_message(
                "user"
            ):

                st.markdown(
                    user_text
                )


            # ---------------------------------------------
            # CURRENT FIELD BEFORE PROCESSING
            # ---------------------------------------------

            current_field = (
                bot.get_next_missing_field(
                    st.session_state.lead
                )
            )


            # ---------------------------------------------
            # BOT RESPONSE
            # ---------------------------------------------

            normalized_user = bot.normalize_for_search(user_text)
            stop_words = {
                "kifayet", "besdir", "dayandir", "dayanin", "stop",
                "istemirem", "lazim deyil", "ehtiyac yoxdur",
            }
            add_to_need_batch = (
                st.session_state.need_batch_active
                and time.time() < st.session_state.need_batch_deadline
                and current_field == "parent_name"
                and normalized_user not in stop_words
            )
            plain_need_input = (
                current_field == "main_concern"
                and normalized_user not in stop_words
                and not user_text.rstrip().endswith("?")
                and not bot.is_clinical_boundary_question(user_text)
            )

            try:

                if add_to_need_batch:
                    child = bot.get_active_child(st.session_state.lead)
                    previous = str(child.get("main_concern") or "").strip()
                    addition = user_text.strip()
                    if addition and addition.lower() not in previous.lower():
                        child["main_concern"] = (
                            f"{previous}; {addition}" if previous else addition
                        )
                        bot.sync_flat_fields(st.session_state.lead)
                    # Pəncərə ilk mesajdan yox, son ehtiyac mesajından sonra
                    # səkkiz saniyə sakitlik keçəndə bağlanır.
                    st.session_state.need_batch_deadline = time.time() + 8.0
                    bot_response = "Əlavə məlumatı da qeyd etdim."
                elif plain_need_input:
                    # Sadə ehtiyac cavabını LLM gözləmədən dərhal state-ə yaz.
                    # Beləliklə paralel rerun yaranmadan debounce aktivləşir.
                    bot.save_current_field_fallback(
                        lead=st.session_state.lead,
                        field="main_concern",
                        user_text=user_text,
                    )
                    st.session_state.need_batch_active = True
                    st.session_state.need_batch_deadline = time.time() + 8.0
                    bot_response = "Anladım, qeyd etdim."
                else:
                    bot_response = (
                        bot.lead_agent_reply(
                        user_text=user_text,
                        lead=st.session_state.lead,
                        faq_min_score=0.25,
                        history=(
                            st.session_state.messages[:-1]
                        ),
                        conversation_id=st.session_state.session_id,
                        channel_message_id=st.session_state.messages[-1]["message_id"],
                            channel="streamlit",
                        )
                    )

                if (
                    not add_to_need_batch
                    and current_field == "main_concern"
                    and bot.get_next_missing_field(st.session_state.lead) == "parent_name"
                    and st.session_state.lead.get("status") != "STOPPED"
                ):
                    st.session_state.need_batch_active = True
                    st.session_state.need_batch_deadline = time.time() + 8.0
                    bot_response = "Anladım, qeyd etdim."

            except Exception as exc:

                bot_response = (
                    "Hazırda texniki problem yarandı. "
                    "Zəhmət olmasa bir qədər sonra "
                    "yenidən cəhd edin."
                )

                print(
                    "BOT ERROR:",
                    exc,
                )


            # ---------------------------------------------
            # SAVE CONVERSATION LOG
            # ---------------------------------------------

            try:

                bot.save_conversation_log(
                    session_id=(
                        st.session_state.session_id
                    ),
                    user_message=user_text,
                    bot_response=bot_response,
                    current_field=current_field,
                    lead=st.session_state.lead,
                )

            except Exception as exc:

                print(
                    "LOG SAVE ERROR:",
                    exc,
                )


            # ---------------------------------------------
            # SHOW BOT RESPONSE
            # ---------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": bot_response,
                }
            )


            with st.chat_message(
                "assistant"
            ):

                st.markdown(
                    bot_response
                )


            # ---------------------------------------------
            # CALL REQUESTED / ESCALATED
            #
            # Hər iki halda lead bazaya yazılır.
            # Əvvəllər yalnız CALL_REQUESTED saxlanılırdı,
            # ona görə eskalasiya olunmuş müraciətlər itirdi.
            # ---------------------------------------------

            if (
                st.session_state.lead.get(
                    "status"
                )
                in (
                    "CALL_REQUESTED",
                    "ESCALATED",
                    "NO_CONTACT",
                )
            ):

                persist_lead_once(
                    st.session_state.lead
                )
                # Database identifier is internal and is not shown to the parent.
                confirmation = None


                if confirmation:

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": confirmation,
                        }
                    )


                    with st.chat_message(
                        "assistant"
                    ):

                        st.markdown(
                            confirmation
                        )


                # Keep chat active for corrections, further questions and handoff.
                st.session_state.conversation_finished = False


# =========================================================
# CONVERSATION FINISHED
# =========================================================

if st.session_state.conversation_finished:

    st.divider()

    st.success(
        "Cari müraciət prosesi tamamlandı."
    )


    if st.button(
        "Yeni test başlat",
        use_container_width=True,
    ):

        reset_chat()

        st.rerun()
