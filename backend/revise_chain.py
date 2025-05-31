import os
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnableMap
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.75,
    top_p=0.8
)

prompt = PromptTemplate.from_template("""
You are a helpful assistant that revises or generates fluent English text.

Task:
- If the input is a complete sentence or paragraph, revise it to match the desired emotion and be appropriate for the intended audience.
- If the input is just a few keywords or phrases, create a short paragraph using them that matches the tone and fits the audience.

Instructions:
- Emotion: {emotion}
- Audience: {audience}
- Input: {text}

Output:
""")

revise_chain = (
    RunnableMap({
        "text": lambda x: x["text"],
        "emotion": lambda x: x["emotion"],
        "audience": lambda x: x["audience"]
    }) 
    | prompt 
    | llm 
    | StrOutputParser()
)
