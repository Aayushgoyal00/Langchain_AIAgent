from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
# import json

OLLAMA_HOST = "http://localhost:11434"

# Initialize the Ollama chat model with TinyLlama
model = ChatOllama(
    model="tinyllama:1.1b",
    base_url=OLLAMA_HOST
)

# Text to summarize
text_to_summarize = "Why Participate? Solve Real Challenges Work on cutting-edge problems in data structures, algorithms, optimization, probability, statistics, calculus, and linear algorithms. Prove Your Skills Compete with the best talent from India s top engineering colleges. Exclusive Career Opportunity Top performers will be fast-tracked to a Pre-Placement Interview (PPI) with Goldman Sachs. Mentorship & Learning Receive mentorship from Goldman Sachs professionals"


schema = [ 
    ResponseSchema(name="name", description="Name of the person"),
    ResponseSchema(name="age", description="Age of the person"),
    ResponseSchema(name="location", description="Location of the person"),
]

parser=StructuredOutputParser.from_response_schemas(schema)
template1=PromptTemplate(
    input_variables=['name'],
    template="Give me a age ,location of a fictional person with name {name}\n{format_instructions}",
    partial_variables={"format_instructions":parser.get_format_instructions()},
)

chain=template1 | model | parser
result=chain.invoke({"name": "Aman"})
print(result)
prompt=template1.invoke({"name": "Aman"})

# result = model.invoke(prompt)
# # print(result)
# final_result = parser.parse(result.content)
# print(final_result)