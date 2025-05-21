from langchain_ollama import ChatOllama
from dotenv import load_dotenv
import os
load_dotenv()

# Connect to Ollama running in Docker
# By default, Ollama API is available at http://localhost:11434
# If your Docker setup exposes a different host or port, adjust accordingly
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Initialize the Ollama chat model with TinyLlama
model = ChatOllama(
    model="tinyllama:1.1b",
    base_url=OLLAMA_HOST
)

# Text to summarize
text_to_summarize = "Why Participate? Solve Real Challenges Work on cutting-edge problems in data structures, algorithms, optimization, probability, statistics, calculus, and linear algorithms. Prove Your Skills Compete with the best talent from India s top engineering colleges. Exclusive Career Opportunity Top performers will be fast-tracked to a Pre-Placement Interview (PPI) with Goldman Sachs. Mentorship & Learning Receive mentorship from Goldman Sachs professionals"

# Create a prompt that explicitly asks for a summary
prompt = f"Please provide a concise summary of the following text in 50 to 100 words in English:\n\n{text_to_summarize}"

# Invoke the model
result = model.invoke(prompt)

# Print the result
print(result)