from google import genai
from google.genai import types
import dotenv
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound,TranscriptsDisabled # Added TranscriptsDisabled and NoTranscriptFound
# Load environment variables from .env file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
dotenv.load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"E:\Langchain\gen-lang-client-0553051082-a99cbe5d72c5.json" # Using raw string for Windows path

video_id = "LVrQcTfm4pc"  # Replace with your YouTube video ID
full_transcript_text = "" # Initialize full_transcript_text
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")


try:
    # List all available transcripts for the video
    list_of_transcripts_obj = YouTubeTranscriptApi.list_transcripts(video_id)
    # print(list_of_transcripts_obj) # Uncomment for debugging, can be very long
    target_transcript_segments = None

    # Strategy to find the best English transcript:
    # 1. Try manually created English transcript
    try:
        transcript = list_of_transcripts_obj.find_manually_created_transcript(['en'])
        target_transcript_segments = transcript.fetch()
        print("Fetched manually created English transcript.")
    except NoTranscriptFound:
        print("No manually created English transcript found. Trying auto-generated.")
        # 2. Try generated English transcript
        try:
            transcript = list_of_transcripts_obj.find_generated_transcript(['en'])
            target_transcript_segments = transcript.fetch()
            # print(target_transcript_segments.to_raw_data())
            print("Fetched auto-generated English transcript.")
        except NoTranscriptFound:
            print("No auto-generated English transcript found. Trying to translate an original.")
            # 3. If no English transcript, fetch the first available original and translate it
            found_original_to_translate = False
            for original_transcript_meta in list_of_transcripts_obj:
                if original_transcript_meta.is_translatable:
                    try:
                        print(f"Attempting to translate original transcript: {original_transcript_meta.language} ({original_transcript_meta.language_code})")
                        translated_transcript_obj = original_transcript_meta.translate('en')
                        target_transcript_segments = translated_transcript_obj.fetch()
                        print(f"Successfully translated and fetched transcript from '{original_transcript_meta.language_code}' to 'en'.")
                        # print(target_transcript_segments.to_raw_data())
                        found_original_to_translate = True
                        break 
                    except Exception as e_translate:
                        print(f"Could not translate transcript from {original_transcript_meta.language_code}: {e_translate}")
            
            if not found_original_to_translate:
                 print("No suitable English transcript found and no original transcript could be translated.")

    if target_transcript_segments:
        full_transcript_text = " ".join([item.text for item in target_transcript_segments])
        # print(full_transcript_text) # Uncomment for debugging, can be very long
        print(f"Successfully processed transcript. Total length: {len(full_transcript_text)} characters.")
        # print(f"Transcript: {full_transcript_text}")
    else:
        print(f"Could not obtain any transcript data for the video {video_id}.")

except TranscriptsDisabled:
    print(f"Transcripts are disabled for video: {video_id}")
except Exception as e:
    print(f"Error fetching transcript: {e}")


splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.create_documents([full_transcript_text])
print(f"Created {len(chunks)} document chunks")

# # Extract the text content from each Document object
# texts = [doc.page_content for doc in chunks]
# vectors = embeddings.embed_documents(texts,output_dimensionality=10)

# print(f"Number of chunks: {chunks}")
# print(f"Number of vectors: {vectors}") 

# Assign unique IDs to each chunk
chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]

# Create FAISS vector store
if chunks:
    # LangChain's FAISS wrapper handles embedding and indexing
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings,
        # ids=chunk_ids  # Assign unique IDs to each document
    )
    print(f"Stored {len(chunks)} vectors in FAISS vector store")
    # print(f"Vector store: {vector_store}")
    # Example: Perform a similarity search
    query = "What is the main topic of the video?"
    # query_embedding = embeddings.embed_query(query)
    # print(f"Query embedding: {query_embedding}")
    # Search for the top 3 most similar chunks
    # results = vector_store.similarity_search_with_score(query=query, k=3)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    results = retriever.invoke(query)
    print(f"Found {results} similar chunks")
    # print("\nTop 3 similar chunks:")
    # for doc, score in results:
    #     print(f"Chunk ID: {doc.id}, Score: {score}")
    #     print(f"Content: {doc.page_content}")  # Print first 200 chars of each chunk
else:
    print("No chunks to store in FAISS.")