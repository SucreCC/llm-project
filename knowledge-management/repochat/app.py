import time
import streamlit as st

from src.utils import init_session_state
from src.git import git_form
from src.db import vector_db, load_to_db
from src.models import hf_embeddings, code_llama
from src.chain import response_chain

# 初始化一些全局变量， 如消息，数据库名称等
init_session_state()

# 为前端页面设置一些配置信息
st.set_page_config(
    page_title="RepoChat",
    page_icon="💻",
    initial_sidebar_state="expanded",
    menu_items={
        'Report a bug': "https://github.com/pnkvalavala/repochat/issues",
        'About': "No need to worry if you can't understand GitHub code or repositories anymore! Introducing RepoChat, where you can effortlessly chat and discuss all things related to GitHub repositories."
    }
)

st.markdown(
    "<h1 style='text-align: center;'>RepoChat</h1>",
    unsafe_allow_html=True
)

try:
    st.session_state["db_name"], st.session_state['git_form'] = git_form(st.session_state['repo_path'])

    if st.session_state['git_form']:
        with st.spinner('Loading the contents to database. This may take some time...'):
            st.session_state["chroma_db"] = vector_db(
                hf_embeddings(),
                load_to_db(st.session_state['repo_path'])
            )
        with st.spinner('Loading model to memory'):
            st.session_state["qa"] = response_chain(
                db=st.session_state["chroma_db"],
                llm=code_llama()
            )

        st.session_state["db_loaded"] = True
except TypeError:
    pass

if st.session_state["db_loaded"]:
    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Enter your query"):
        st.session_state["messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            with st.spinner("Generating response..."):
                result = st.session_state["qa"](prompt)
            for chunk in result['answer'].split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
        st.session_state["messages"].append({"role": "assistant", "content": full_response})