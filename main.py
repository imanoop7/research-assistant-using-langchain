from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
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

output=RunnablePassthrough.assign(
    urls = lambda x: web_search(x["question"])
)| (lambda x: [{"question":x["question"], "url":u}for u in x["urls"]])|chain.map()



response =output.invoke({
    "question" : "what is langgraph ?"
})

