import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"E:\Langchain\gen-lang-client-0553051082-a99cbe5d72c5.json" # Using raw string for Windows path

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
vector = embeddings.embed_query("hello, world!")
print(len(vector))
