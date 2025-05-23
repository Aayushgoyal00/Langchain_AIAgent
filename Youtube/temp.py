from google import genai
from google.genai import types
import dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound,TranscriptsDisabled # Added TranscriptsDisabled and NoTranscriptFound
# Load environment variables from .env file
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableParallel,RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
dotenv.load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"E:\Langchain\gen-lang-client-0553051082-a99cbe5d72c5.json" # Using raw string for Windows path

video_id = "LVrQcTfm4pc"  # Replace with your YouTube video ID
full_transcript_text = "" # Initialize full_transcript_text
# embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-exp-03-07")
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
prompt=PromptTemplate(
    input_variables=["context", "question"],
    template="You are an assistant." \
    "Answer only from the given context and if the context given to you is insufficient then just say you don't know." \
    "Context: {context}" \
    "Question: {question}",
)
parser=StrOutputParser()
def format_docs(docs):
    # Format the documents for output
    context="\n\n".join([doc.page_content for doc in docs])
    return context

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



chunks = splitter.create_documents([full_transcript_text])
print(f"Created {len(chunks)} document chunks")

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

    question = "Is there any discussion about the deep learning in this context?"
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    # results = retriever.invoke(question)
    # print(f"Found {results} similar chunks")
    # context="\n\n".join([doc.page_content for doc in results])

    # final_prompt = prompt.invoke({"context": context, "question": question})
    # print(f"Final Prompt: {final_prompt}")
    # answer = llm.invoke(final_prompt)
    # print(f"Answer: {answer}")


    parallel_chain=RunnableParallel({
        'context':retriever|RunnableLambda(format_docs),
        'question':RunnablePassthrough()
    })

    main_chain=parallel_chain |prompt| llm|parser
    final_result=main_chain.invoke('does this context contain any discussion about the working of carry minati?')
    print(f"Final Result: {final_result}")

else:
    print("No chunks to store in FAISS.")