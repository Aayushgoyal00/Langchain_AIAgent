# from langchain_huggingface import HuggingFaceEmbeddings
# embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
import os
load_dotenv()
print("API Token:", os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"))
llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN"),
)
model = ChatHuggingFace(llm=llm)

while True:
    query=input("You: ")
    if query.lower() =="exit":
        break
    result = model.invoke(query)
    print("AI :",result)
# result = embedding.embed_query("hello world")
# print(result)