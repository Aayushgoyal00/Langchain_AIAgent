from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
OLLAMA_HOST = "http://localhost:11434"

# Initialize the Ollama chat model with TinyLlama
model = ChatOllama(
    model="tinyllama:1.1b",
    base_url=OLLAMA_HOST
)
prompt1=PromptTemplate(
    input_variables=["topic"],
    template="Please provide a concise summary on the {topic} in 100 words in English.",
)
prompt2=PromptTemplate(
    input_variables=["topic"],
    template="Please provide a 5 points summary on the {topic} in English.",
)
parser=StrOutputParser()
chain=prompt1 | model | parser | prompt2 | model |parser
result=chain.invoke({"topic": "Microsoft"})
print(result)