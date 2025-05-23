from google import genai
from google.genai import types
import dotenv
import os
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled # Added TranscriptsDisabled and NoTranscriptFound
# Load environment variables from .env file
dotenv.load_dotenv()
client = genai.Client(api_key=os.getenv("GENAI_API_KEY"))

video_id = "2iaPZQ4PBpc"  # Replace with your YouTube video ID
full_transcript_text = "" # Initialize full_transcript_text

try:
    # List all available transcripts for the video
    list_of_transcripts_obj = YouTubeTranscriptApi.list_transcripts(video_id)

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
        print(f"Transcript: {full_transcript_text}")
    else:
        print(f"Could not obtain any transcript data for the video {video_id}.")

except TranscriptsDisabled:
    print(f"Transcripts are disabled for video: {video_id}")
except Exception as e:
    print(f"Error fetching transcript: {e}")

# response = client.models.generate_content(
#     model="gemini-2.0-flash",
#     contents=[f"Summarise {full_transcript_text} in 200 words"],
#     config=types.GenerateContentConfig(
#         max_output_tokens=500,
#         temperature=1
#     )
# )
# print(response.text)