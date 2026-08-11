import uuid
import sqlite3

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
    "Salam, xoş gəlmisiniz 🙏 "
    "Junior Coaching proqramına maraq göstərdiyiniz üçün "
    "təşəkkür edirik. "
    "Sizə necə müraciət edə bilərəm?"
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

    # Hələlik test üçün.
    # Sonra bunu Streamlit Secrets-ə keçirəcəyik.
    ADMIN_PASSWORD = "junior2026"

    if admin_password:

        if admin_password == ADMIN_PASSWORD:

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
# CHAT INPUT
# =========================================================

if not st.session_state.conversation_finished:

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

            try:

                bot_response = (
                    bot.lead_agent_reply(
                        user_text=user_text,
                        lead=st.session_state.lead,
                        faq_min_score=0.25,
                    )
                )

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
            # CALL REQUESTED
            # ---------------------------------------------

            if (
                st.session_state.lead.get(
                    "status"
                )
                == "CALL_REQUESTED"
            ):

                if not st.session_state.lead_saved:

                    phone = (
                        st.session_state.lead.get(
                            "phone"
                        )
                    )


                    existing_lead = (
                        bot.find_lead_by_phone(
                            phone
                        )
                    )


                    if existing_lead is None:

                        try:

                            lead_id = (
                                bot.save_lead_to_db(
                                    st.session_state.lead
                                )
                            )


                            confirmation = (
                                "Müraciətiniz sistemdə "
                                "qeyd edildi. "
                                f"Müraciət nömrəniz: {lead_id}"
                            )

                        except Exception as exc:

                            print(
                                "LEAD SAVE ERROR:",
                                exc,
                            )

                            confirmation = (
                                "Müraciət məlumatlarınız "
                                "toplandı, lakin sistemdə "
                                "saxlanarkən texniki problem yarandı."
                            )

                    else:

                        confirmation = (
                            "Bu telefon nömrəsi ilə artıq "
                            f"{existing_lead['id']} nömrəli "
                            "müraciət mövcuddur. "
                            "Məlumatınız komanda tərəfindən "
                            "nəzərə alınacaq."
                        )


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


                    st.session_state.lead_saved = True


                st.session_state.conversation_finished = True


            # ---------------------------------------------
            # ESCALATED
            # ---------------------------------------------

            elif (
                st.session_state.lead.get(
                    "status"
                )
                == "ESCALATED"
            ):

                st.session_state.conversation_finished = True


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