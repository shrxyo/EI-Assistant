import streamlit as st
from openai import OpenAI

client = OpenAI(api_key="sk..")
model_name = "ft:gpt-4o-mini-2024-07-18:hawshiuan:therapy-bot:BU1Z7786"

st.title("Therapy Chatbot")
st.write("You are chatting with a helpful and joyous mental therapy assistant.")

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful and joyous mental therapy assistant."}
    ]

with st.form(key="chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", key="user_input")
    submitted = st.form_submit_button("Send")
    if submitted and user_input.strip():
        st.session_state['messages'].append({"role": "user", "content": user_input})
        response = client.chat.completions.create(
            model=model_name,
            messages=st.session_state['messages'],
            max_tokens=200
        )
        assistant_message = response.choices[0].message.content
        st.session_state['messages'].append({"role": "assistant", "content": assistant_message})

for msg in st.session_state['messages'][1:]:
    if msg['role'] == 'user':
        st.markdown(f"**You:** {msg['content']}")
    elif msg['role'] == 'assistant':
        st.markdown(f"**Bot:** {msg['content']}")