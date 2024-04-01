from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import requests
from bs4 import BeautifulSoup


template = """Summarize the data based on the context{contex}
Question{question}
"""


promt= ChatPromptTemplate.from_template(template)

url="https://blog.langchain.dev/code-execution-with-langgraph/"

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