import os
import dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled

# Load environment variables
dotenv.load_dotenv()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = r"E:\Langchain\gen-lang-client-0553051082-a99cbe5d72c5.json"

video_id = "EzYaFF7ahKw"  # Replace with your YouTube video ID
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.2,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Function to format timestamps from seconds to MM:SS
def format_time(seconds):
    minutes = int(seconds // 60)
    seconds = int(seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"

# Function to group transcript segments into chunks
def group_segments(segments, max_length=300):
    chunks = []
    current_chunk = []
    current_length = 0
    
    for segment in segments:
        segment_text = segment.text  # Use dot notation for FetchedTranscriptSnippet
        segment_length = len(segment_text)
        
        if current_length + segment_length > max_length and current_chunk:
            chunks.append(current_chunk)
            current_chunk = [segment]
            current_length = segment_length
        else:
            current_chunk.append(segment)
            current_length += segment_length
    
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Function to format retrieved documents
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

try:
    # Fetch transcripts
    list_of_transcripts_obj = YouTubeTranscriptApi.list_transcripts(video_id)
    target_transcript_segments = None

    # Try manually created English transcript
    try:
        transcript = list_of_transcripts_obj.find_manually_created_transcript(['en'])
        target_transcript_segments = transcript.fetch()
        print("Fetched manually created English transcript.")
    except NoTranscriptFound:
        # Try auto-generated English transcript
        try:
            transcript = list_of_transcripts_obj.find_generated_transcript(['en'])
            target_transcript_segments = transcript.fetch()
            print("Fetched auto-generated English transcript.")
        except NoTranscriptFound:
            # Try translating an available transcript
            for original_transcript_meta in list_of_transcripts_obj:
                if original_transcript_meta.is_translatable:
                    try:
                        translated_transcript_obj = original_transcript_meta.translate('en')
                        target_transcript_segments = translated_transcript_obj.fetch()
                        print(f"Translated transcript from '{original_transcript_meta.language_code}' to 'en'.")
                        break
                    except Exception as e:
                        print(f"Could not translate from {original_transcript_meta.language_code}: {e}")
            if target_transcript_segments is None:
                print("No suitable English transcript found.")

    if target_transcript_segments:
        # Create documents with timestamps
        documents = []
        # print(target_transcript_segments)
        for chunk in group_segments(target_transcript_segments):
            if chunk:
                start_time = chunk[0].start
                end_time = chunk[-1].start + chunk[-1].duration
                time_range = f"{format_time(start_time)} - {format_time(end_time)}"
                text = " ".join([seg.text for seg in chunk])
                page_content = f"[{time_range}] {text}"
                doc = Document(page_content=page_content)
                documents.append(doc)
        print(f"Created {len(documents)} document chunks")
        # print(documents) 
        # Create FAISS vector store
        vector_store = FAISS.from_documents(documents, embeddings)
        print(f"Stored {len(documents)} vectors in FAISS vector store")

        # Set up retriever
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

        # Updated prompt to include timestamp instructions
        # prompt = PromptTemplate(
        #     input_variables=["context", "question"],
        #     template="You are an assistant. Answer the question based on the following context, which includes timestamps and transcript text in the format [MM:SS - MM:SS]. Include the relevant timestamps in your answer. If the context is insufficient to answer the question, say 'I don't know' . List all relevant timestamps if the answer appears in multiple places.\n\nContext: {context}\n\nQuestion: {question}",
        # )
        prompt=PromptTemplate(
        input_variables=["context", "question"],
        template="You are an assistant." \
        "Answer only from the given context and if the context given to you is insufficient then just say you don't know." \
        "Context: {context}" \
        "Question: {question}",)
        # Set up processing chain
        parallel_chain = RunnableParallel({
            'context': retriever | RunnableLambda(format_docs),
            'question': RunnablePassthrough()
        })
        
        main_chain = parallel_chain | prompt | llm | StrOutputParser()
        
        # Example question
        final_result = main_chain.invoke('what are the topics discussed in the video ?')
        print(f"Final Result: {final_result}")

    else:
        print(f"Could not obtain transcript data for video {video_id}.")

except TranscriptsDisabled:
    print(f"Transcripts are disabled for video: {video_id}")
except Exception as e:
    print(f"Error fetching transcript: {e}")