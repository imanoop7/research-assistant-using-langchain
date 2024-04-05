from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
import json
load_dotenv()



os.environ["LANGCHAIN_API_KEY"] = str(os.getenv("LANGCHAIN_API_KEY"))
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "TEST LLM"

RESULTS_PER_QUESTION=3
search = DuckDuckGoSearchAPIWrapper()
def web_search(query, num_results=RESULTS_PER_QUESTION):
    results = search.results(query, num_results)
    return [r['link'] for r in results]


template = """Summarize the data based on the context{context}
Question{question}
"""

google_api_key = os.getenv("GOOGLE_API_KEY")
model = ChatGoogleGenerativeAI(model='gemini-pro', google_api_key=google_api_key)
prompt= ChatPromptTemplate.from_template(template)


def  web_scraper(url):
    try: 
        response = requests.get(url)

        if response.status_code ==200:
            soup= BeautifulSoup(response.text, "html.parser")

            page_text = soup.get_text(separator=" ", strip= True)

            return page_text
        else:
           return "Failed to retrieve the page Status code {response.status_code}"
    except Exception as e:
        print(e)
        return f"Failed to retrive the page{e}"
    


# url="https://blog.langchain.dev/code-execution-with-langgraph/"


chain = RunnablePassthrough.assign(
    context= lambda x:web_scraper(x["url"])[:10000]
)|prompt|model|StrOutputParser()

output_web_chain=RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
)| (lambda x: [{"question":x["question"], "url":u}for u in x["urls"]])|chain.map()



# response =output_web_chain.invoke({
#     "question" : "what is langgraph ?"
# })


SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user",
            "Write 3 google search queries to search online that form an "
            "objective opinion from the following: {question}\n"
            "You must respond with a list of strings in the following format: "
            '["query 1", "query 2", "query 3"].',
        ),
    ]
)

search_question_chain = SEARCH_PROMPT | model | StrOutputParser() | json.loads

full_research_chain = search_question_chain | (lambda x: [{"question": q} for q in x]) | output_web_chain.map()

WRITER_SYSTEM_PROMPT = "You are an AI critical thinker research assistant. Your sole purpose is to write well written, critically acclaimed, objective and structured reports on given text."  # noqa: E501



RESEARCH_REPORT_TEMPLATE = """Information:
--------
{research_summary}
--------
Using the above information, answer the following question or topic: "{question}" in a detailed report -- \
The report should focus on the answer to the question, should be well structured, informative, \
in depth, with facts and numbers if available and a minimum of 1,200 words.
You should strive to write the report as long as you can using all relevant and necessary information provided.
You must write the report with markdown syntax.
You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.
Write all used source urls at the end of the report, and make sure to not add duplicated sources, but only one reference for each.
You must write the report in apa format.
Please do your best, this is very important to my career."""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WRITER_SYSTEM_PROMPT),
        ("user", RESEARCH_REPORT_TEMPLATE),
    ]
)

def collapse_list_of_lists(list_of_lists):
    content = []
    for l in list_of_lists:
        content.append("\n\n".join(l))
    return "\n\n".join(content)

chain_last = RunnablePassthrough.assign(
    research_summary= full_research_chain | collapse_list_of_lists
) | prompt | model | StrOutputParser()




#!/usr/bin/env python
from fastapi import FastAPI
from langserve import add_routes


app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple api server using Langchain's Runnable interfaces",
)

add_routes(
    app,
    chain_last,
    path="/research-assistant",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
