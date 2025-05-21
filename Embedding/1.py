# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv

# load_dotenv()
# embedding=OpenAIEmbeddings(model="text-embedding-ada-002", dimensions=32)
# result=embedding.embed_query("hello world")
# print(result)

from langchain_huggingface import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
result = embedding.embed_query("hello world")
print(result)