import uuid
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
# SESSION INITIALIZATION
# =========================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


if "lead" not in st.session_state:
    st.session_state.lead = bot.create_empty_lead(
        source="Streamlit"
    )


if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Salam, xoş gəlmisiniz 🙏 "
                "Junior Coaching proqramına maraq göstərdiyiniz "
                "üçün təşəkkür edirik. "
                "Sizə necə müraciət edə bilərəm?"
            ),
        }
    ]


if "conversation_finished" not in st.session_state:
    st.session_state.conversation_finished = False


# =========================================================
# TITLE
# =========================================================

st.title("Junior Coaching")

st.caption(
    "Junior Coaching proqramı üzrə virtual müraciət köməkçisi"
)


# =========================================================
# SHOW CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):
        st.markdown(
            message["content"]
        )


# =========================================================
# USER INPUT
# =========================================================

if not st.session_state.conversation_finished:

    user_text = st.chat_input(
        "Mesajınızı yazın..."
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

        with st.chat_message("user"):
            st.markdown(user_text)


        # ---------------------------------------------
        # CURRENT FIELD
        # ---------------------------------------------

        current_field = bot.get_next_missing_field(
            st.session_state.lead
        )


        # ---------------------------------------------
        # BOT RESPONSE
        # ---------------------------------------------

        bot_response = bot.lead_agent_reply(
            user_text=user_text,
            lead=st.session_state.lead,
            faq_min_score=0.25,
        )


        # ---------------------------------------------
        # SAVE CONVERSATION LOG
        # ---------------------------------------------

        bot.save_conversation_log(
            session_id=st.session_state.session_id,
            user_message=user_text,
            bot_response=bot_response,
            current_field=current_field,
            lead=st.session_state.lead,
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

        with st.chat_message("assistant"):
            st.markdown(bot_response)


        # ---------------------------------------------
        # LEAD COMPLETED
        # ---------------------------------------------

        if (
            st.session_state.lead.get("status")
            == "CALL_REQUESTED"
        ):

            phone = st.session_state.lead.get(
                "phone"
            )

            existing_lead = bot.find_lead_by_phone(
                phone
            )

            if existing_lead is None:

                lead_id = bot.save_lead_to_db(
                    st.session_state.lead
                )

                confirmation = (
                    f"Müraciətiniz sistemdə qeyd edildi. "
                    f"Müraciət nömrəniz: {lead_id}"
                )

            else:

                confirmation = (
                    "Bu telefon nömrəsi ilə artıq "
                    f"{existing_lead['id']} nömrəli müraciət "
                    "mövcuddur. Məlumatınız komanda tərəfindən "
                    "nəzərə alınacaq."
                )


            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": confirmation,
                }
            )

            with st.chat_message("assistant"):
                st.markdown(
                    confirmation
                )


            st.session_state.conversation_finished = True


        elif (
            st.session_state.lead.get("status")
            == "ESCALATED"
        ):

            st.session_state.conversation_finished = True


# =========================================================
# NEW TEST BUTTON
# =========================================================

if st.session_state.conversation_finished:

    st.divider()

    if st.button(
        "Yeni test başlat"
    ):

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
                "content": (
                    "Salam, xoş gəlmisiniz 🙏 "
                    "Junior Coaching proqramına maraq göstərdiyiniz "
                    "üçün təşəkkür edirik. "
                    "Sizə necə müraciət edə bilərəm?"
                ),
            }
        ]

        st.session_state.conversation_finished = False

        st.rerun()