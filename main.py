import os
from dotenv import load_dotenv
load_dotenv()
import assemblyai as aai
import cohere
from moviepy import VideoFileClip 
from pydub import AudioSegment
from pydub.effects import normalize
import base64

# Initialize API clients using environment variables
aai.settings.api_key = os.getenv("ASSEMBLYAI_API_KEY")
cohere_client = cohere.Client(os.getenv("COHERE_API_KEY"))

def extract_audio_from_video(file_path):
    """Extracts and normalizes audio from the video and saves it as a .wav file."""
    video = VideoFileClip(file_path)
    audio = video.audio
    audio_file_path = "temp_audio.wav"
    audio.write_audiofile(audio_file_path, codec='pcm_s16le')

    # Normalize the audio
    sound = AudioSegment.from_wav(audio_file_path)
    normalized_sound = normalize(sound)
    normalized_audio_file_path = "normalized_temp_audio.wav"
    normalized_sound.export(normalized_audio_file_path, format="wav")

    return normalized_audio_file_path

def diarize_audio(audio_file_path):
    """Diarizes the audio file using AssemblyAI with clear formatting."""
    config = aai.TranscriptionConfig(speaker_labels=True)
    transcriber = aai.Transcriber()
    transcript = transcriber.transcribe(audio_file_path, config)

    diarized_segments = []
    for utterance in transcript.utterances:
        start_time = utterance.start / 1000  # Convert milliseconds to seconds
        end_time = utterance.end / 1000      # Convert milliseconds to seconds
        formatted_time = f"[{start_time:.2f}s - {end_time:.2f}s]"
        diarized_segments.append(f"{formatted_time} Speaker {utterance.speaker}: {utterance.text}")

    return "\n\n".join(diarized_segments), transcript.text

def summarize_conversation(conversation_text):
    """Generates a summary using Cohere based on the entire conversation text."""
    prompt = f"Summarize the following conversation:\n\n{conversation_text}\n\nSummary:"
    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop_sequences=["\n"],
        return_likelihoods='NONE'
    )
    return response.generations[0].text.strip()

def generate_answer(document_text, question):
    """Generates an answer using Cohere based on the given document text and question."""
    prompt = f"Based on the following document, answer the question:\n\nDocument:\n{document_text}\n\nQuestion: {question}\n\nAnswer:"
    response = cohere_client.generate(
        model='command-xlarge-nightly',
        prompt=prompt,
        max_tokens=150,
        temperature=0.7,
        stop_sequences=["\n"],
        return_likelihoods='NONE'
    )
    return response.generations[0].text.strip()

def generate_download_link(text, filename):
    """Generates a download link for the given text."""
    b64 = base64.b64encode(text.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download {filename}</a>'
