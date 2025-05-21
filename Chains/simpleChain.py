from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
OLLAMA_HOST = "http://localhost:11434"

# Initialize the Ollama chat model with TinyLlama
model = ChatOllama(
    model="tinyllama:1.1b",
    base_url=OLLAMA_HOST
)

prompt=PromptTemplate(
    input_variables=["topic"],
    template="Please provide a 5 facts about {topic} in English.",
)

parsor=StrOutputParser()
chain=prompt | model | parsor
result=chain.invoke({"topic": "Python"})
print(result)
# chain.get_graph().print_ascii()