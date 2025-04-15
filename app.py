import streamlit as st
import os

from main import extract_audio_from_video, diarize_audio, summarize_conversation, generate_answer, generate_download_link

st.title("Video/Audio Diarization, Summary, and Q&A")

# Sidebar options for displaying outputs
st.sidebar.header("Display Options")
show_diarization = st.sidebar.checkbox("Show Diarization", True)
show_transcript = st.sidebar.checkbox("Show Full Transcript", False)
show_summary = st.sidebar.checkbox("Show Summary", True)

# Create the "uploads" directory if it doesn't exist
os.makedirs("uploads", exist_ok=True)

# File upload
uploaded_file = st.file_uploader("Upload MP4/MP3 file", type=["mp4", "mp3"])
if uploaded_file is not None:
    # Save uploaded file
    file_path = os.path.join("uploads", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    file_extension = uploaded_file.name.split('.')[-1].lower()

    # Extract audio if file is a video (mp4); use directly if audio (mp3)
    if file_extension == "mp4":
        with st.spinner("Extracting audio from video..."):
            audio_file_path = extract_audio_from_video(file_path)
    else:
        audio_file_path = file_path

    # Diarize audio
    with st.spinner("Processing diarization..."):
        diarization_output, full_text = diarize_audio(audio_file_path)

    if show_diarization:
        st.subheader("Diarization Output")
        st.text_area("Diarization Result", diarization_output, height=300)

    # Download link for full transcript
    st.markdown(generate_download_link(full_text, "full_transcript.txt"), unsafe_allow_html=True)
    if show_transcript:
        st.subheader("Full Transcript")
        st.text_area("Transcript", full_text, height=300)

    # Generate and display summary
    with st.spinner("Generating summary..."):
        summary = summarize_conversation(full_text)
    if show_summary:
        st.subheader("Summary of the Conversation")
        st.text_area("Summary", summary, height=150)

    # Download link for summary
    st.markdown(generate_download_link(summary, "summary.txt"), unsafe_allow_html=True)

    # Question answering section
    question = st.text_input("Ask a question about the content:")
    if question:
        with st.spinner("Generating answer..."):
            answer = generate_answer(full_text, question)
        st.subheader("Answer")
        st.write(answer)

    # Reset button to restart the app
    if st.button("Reset"):
        st.experimental_rerun()
