# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: langchain
#     language: python
#     name: python3
# ---

# Install libraries:
#
# 1) Install miniconda from the website
# 2) Add conda to PATH:
#      - Start Menu and search for "Environment Variables."
#      - Click on "Edit the system environment variables."
#      - In the System Properties window, click on "Environment Variables."
#      - In the Environment Variables window, find and select the Path.
#      - Add new: C:\ProgramData\miniconda3\Scripts (or where yours is located).
#
# 3) Install libs:
#
# Create a new virtual environment named 'langchain' with Python 3.12
# ```
# conda create -n langchain python=3.12 -y
# ```
# List all environments (the active one will be marked with an asterisk)
# ```
# conda env list
# ```
# In PowerShell (admin): Set the execution policy to RemoteSigned for the current user
# ```
# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
# ```
#
# ```
# conda activate langchain
# pip install langchain langchain_openai langchain_community langgraph ipykernel python-dotenv
# ```
#
# If you encounter a dependency resolver error with pip:
# ```
# pip install google-auth # Must be <3.0.dev0,>=2.14.1
# ```
#
# ```
# ipython kernel install --user --name=langchain
# pip install google-search-results
# pip install matplotlib
# pip install wikipedia
# ```

# +
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.agent_toolkits.load_tools import get_all_tool_names

# +
# load_tools() is only a shorthand function. It's better to properly use langchain_community.tools
all_tool_names = get_all_tool_names()
print('length: ', len(all_tool_names))

for i in range(0, len(all_tool_names), 10):
    print(" ".join(all_tool_names[i:i+10]))
# -

# Types of LLMs:
# - Language models - input stings and generate strings. They are typically older and work best to answer individual user queries.
# - Chat model - inputs a sequence of messages, and generates responses that are contextually aware of the conversation flow. By default does not remember past conversations (stateless), but sometimes conversational memory is implemented. 
# - Instruct models - optimized to follow specific instructions and perform tasks rather than engage in open-ended dialogue. 

# +
# Chat model

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

chat_model = ChatOpenAI(api_key=api_key, model='gpt-4o-mini')


messages = [SystemMessage(content='You are a Scientist in AI.'),                # instruct a system to behave in certain way: tone, rules, how respond
            HumanMessage(content="Which are the most 3 hot topic in AI now?")]  # input or queries made by the user/human    

print(chat_model.invoke(messages).content)
# -

# #### Tools are classes that an Agent uses to interact with the world. https://python.langchain.com/api_reference/community/tools.html 

# +
# Google jobs tool

from langchain_community.tools.google_jobs import GoogleJobsQueryRun
from langchain_community.utilities.google_jobs import GoogleJobsAPIWrapper

api_key_serpapi = os.getenv('SERPAPI_API_KEY')
os.environ["SERPAPI_API_KEY"] = api_key_serpapi

# Take special care of description
description_googlejobs = """
A wrapper around Google Jobs Search. 
Useful for when you need to get information aboutgoogle search Jobs from Google JobsInput should be a search query. 
"""

googlejobs = GoogleJobsQueryRun(api_wrapper=GoogleJobsAPIWrapper(), 
                                     description=description_googlejobs)

# Example usage or run(): 
response_g = googlejobs.run("Can I get a list of 3 job posting related to machine learning in alberta")
print(response_g[:1000])
# -

# Golden query tool. 10 free queries
if False: 
    import json
    from langchain_community.utilities.golden_query import GoldenQueryAPIWrapper

    api_key_golden = os.getenv('GOLDEN_API_KEY')
    golden_query = GoldenQueryAPIWrapper()

    json.loads(golden_query.run("companies in nanotech"))

# +
from langchain_community.tools import WikipediaQueryRun  
from langchain_community.utilities import WikipediaAPIWrapper

description_wiki = """
A tool to explain things in text format. Use this tool if you think the user  asked concept is best explained through text.
""" 

wiki_api_wrapper = WikipediaAPIWrapper(doc_content_chars_max=280)
wikipedia = WikipediaQueryRun(description=description_wiki,
                               api_wrapper=wiki_api_wrapper)
# -

tools = [googlejobs, wikipedia]

# - This agent (langchain.agents.react.agent.create_react_agent) from LangChain - is not good 
# - Better use from the LangGraph (offers a more flexible and full-featured)
#
# There are agent types in LangChain:
#
# - React agents: Use reasoning and actions to decide on the best steps 
# - Multi-agent: Divide complicated problems into units of work that can be targeted by specialized agents 
# - Conversational agents: Engage in dialogue and maintain context across multiple interactions 
# - Structured chat agents: Parse inputs and outputs into structured formats 
# - Tool calling agents: Use tools in a straightforward way 
# - Self-ask with search: Split queries into smaller steps to handle them 

# +
from langgraph.prebuilt import create_react_agent
 
prompt_system = SystemMessage("You are a recruiter, who is trying to help me in finding suitable jobs")
prompt_human  = HumanMessage("Find me the most recent Machine Learning job in Alberta, specify when the job was posted")

agent = create_react_agent(chat_model,
                           tools,
                           state_modifier=prompt_system)

response = agent.invoke({"messages": prompt_human})
# -

print(response['messages'][3].content)

response['messages']

response['messages'][1].tool_calls[0]['name']

# +
prompt_system = SystemMessage("You are my assistant")
prompt_human  = HumanMessage("What is Amii (research institute)?")

agent = create_react_agent(chat_model,
                           tools,
                           state_modifier=prompt_system)

response1 = agent.invoke({"messages": prompt_human})
print(response1['messages'][3].content)

# +
# We now see that it is correctly using wikipedia tool to answer my question

response1['messages'][1].tool_calls[0]['name']
# -

response1


