from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
import json

OLLAMA_HOST = "http://localhost:11434"

# Initialize the Ollama chat model with TinyLlama
model = ChatOllama(
    model="tinyllama:1.1b",
    base_url=OLLAMA_HOST
)

# Text to summarize
text_to_summarize = "Why Participate? Solve Real Challenges Work on cutting-edge problems in data structures, algorithms, optimization, probability, statistics, calculus, and linear algorithms. Prove Your Skills Compete with the best talent from India s top engineering colleges. Exclusive Career Opportunity Top performers will be fast-tracked to a Pre-Placement Interview (PPI) with Goldman Sachs. Mentorship & Learning Receive mentorship from Goldman Sachs professionals"

parser=JsonOutputParser()
template1=PromptTemplate(
    input_variables=[],
    template="Give me a name, age ,location of a fictional person\n{format_instructions}",
    partial_variables={"format_instructions":parser.get_format_instructions()},
)
# prompt=template1.invoke({})
# result = model.invoke(prompt.text)
# final_result = parser.parse(result.content) 
# json_string = json.dumps(final_result)
# print(json_string)

chain=template1 | model | parser
result=chain.invoke({})
print(result)


# print(final_result)
# print(result.content)   
# print(prompt)
# template1=PromptTemplate(
#     input_variables=["text"],
#     template="Please provide a concise summary of the following text in 100 words in English:\n\n{text}",
# )
# template2=PromptTemplate(
#     input_variables=["text2"],
#     template="Please provide a 5 points summary of the following text in English:\n\n{text2}",
# )


# parser=StrOutputParser()
# chain=template1 | model | parser | template2 | model |parser
# result=chain.invoke({"text": text_to_summarize})
# print(result)