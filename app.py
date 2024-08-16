import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

## Arxiv and Wikipedia Tools
arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

search = DuckDuckGoSearchRun(name="Search")

# CSS for heading and input field
st.markdown("""
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css">

    <style>
    .header {
        font-size: 2em;
        font-weight: bold;
        text-align: center;
        transition: color 0.3s, transform 0.3s, background-color 0.3s, border-color 0.3s;
        border: 2px solid white;
        padding: 10px;
        background-color: black;
        margin-bottom: 40px;
    }
    .header:hover {
        color: red;
        transform: scale(1.1);
        background-color: lightgrey;
        border-color: red;
    }
    .search-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-top: 20px;
    }
    .search-container input {
        flex: 1;
        padding: 10px;
        font-size: 1em;
    }
    .search-container button {
        padding: 10px;
        font-size: 1em;
        background-color: #00f;
        color: white;
        border: none;
        cursor: pointer;
    }
    .search-container button:hover {
        background-color: #00c;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<div class="header">Chat <i class="fa-regular fa-comment-dots"></i> <span style="margin-left:8px; margin-right:8px;">&</span> Search <i class="fa-brands fa-searchengin"></i> </div>', unsafe_allow_html=True)

## Sidebar for settings 
st.sidebar.markdown("""
    <span style="font-size: 30px; color: red">Chat & Search</span>
    <div style="display: flex; align-items: center; padding-bottom: 20px;">
        <i class="fas fa-cog" style="color: white; font-size: 24px;"></i>
        <span style="margin-left: 8px; font-size: 25px;">Settings</span>
    </div>
    """, unsafe_allow_html=True)

api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web for you. How can I help you?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# Add the search input form
with st.form(key='search_form'):
    search_container = st.container()
    with search_container:
        col1, col2 = st.columns([4, 1])
        with col1:
            prompt = st.text_input("Message ChatBOT...", placeholder="Type your message here")
        with col2:
            search_button = st.form_submit_button("Search")

if search_button and prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
    tools = [search, arxiv, wiki]

    search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, handle_parsing_errors=True)

    with st.chat_message("assistant"):
        st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
        with st.spinner('Searching...'):
            try:
                response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
                st.session_state.messages.append({'role': 'assistant', "content": response})
                st.write(response)
            except ValueError as e:
                st.session_state.messages.append({'role': 'assistant', "content": str(e)})
                st.write(f"An error occurred: {str(e)}")


# import streamlit as st
# from langchain_groq import ChatGroq
# from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
# from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain.agents import initialize_agent, AgentType
# from langchain.callbacks import StreamlitCallbackHandler
# import os
# from dotenv import load_dotenv

# ## Arxiv and Wikipedia Tools
# arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)

# api_wrapper = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=200)
# wiki = WikipediaQueryRun(api_wrapper=api_wrapper)

# search = DuckDuckGoSearchRun(name="Search")

# st.title("🔎 LangChain - Chat with search")

# ## Sidebar for settings
# st.sidebar.title("Settings")
# api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [
#         {"role": "assistant", "content": "Hi, I'm a chatbot who can search the web. How can I help you?"}
#     ]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg['content'])

# if prompt := st.chat_input(placeholder="Message ChatBOT..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     llm = ChatGroq(groq_api_key=api_key, model_name="Llama3-8b-8192", streaming=True)
#     tools = [search, arxiv, wiki]

#     search_agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

#     with st.chat_message("assistant"):
#         st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False)
#         try:
#             response = search_agent.run(st.session_state.messages, callbacks=[st_cb])
#             st.session_state.messages.append({'role': 'assistant', "content": response})
#             st.write(response)
#         except ValueError as e:
#             st.session_state.messages.append({'role': 'assistant', "content": str(e)})
#             st.write(f"An error occurred: {str(e)}")


