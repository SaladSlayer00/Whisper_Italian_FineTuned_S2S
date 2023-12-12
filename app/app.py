import gradio as gr
import pytube as pt
from transformers import pipeline
import os
from huggingface_hub import HfFolder
from gtts import gTTS
from fpdf import FPDF
from pdfminer.high_level import extract_text

# Initialize pipelines for transcription, summarization, and translation
transcription_pipe = pipeline(model="SaladSlayer00/another_local", token=HfFolder.get_token())
summarizer = pipeline("summarization", model="it5/it5-efficient-small-el32-news-summarization")
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-it-en")


def process_audio(file_path):
    text = transcription_pipe(file_path)["text"]
    summary = summarizer(text, min_length=25, max_length=50)[0]["summary_text"]
    translation = translator(text)[0]["translation_text"]
    return text, summary, translation


def download_youtube_audio(yt_url):
    yt = pt.YouTube(yt_url)
    stream = yt.streams.filter(only_audio=True).first()
    file_path = stream.download(filename="temp_audio.mp3")
    return file_path


def youtube_transcription(yt_url):
    audio_path = download_youtube_audio(yt_url)
    results = process_audio(audio_path)
    os.remove(audio_path)  # Clean up the downloaded file
    return results


def transcribe_and_process(rec=None, file=None):
    if rec is not None:
        audio = rec
    elif file is not None:
        audio = file
    else:
        return "Provide a recording or a file."

    return process_audio(audio)


def save_text_to_pdf(text, filename="output.pdf"):
    # Create instance of FPDF class
    pdf = FPDF()

    # Add a page
    pdf.add_page()

    # Set font: Arial, bold, 12
    pdf.set_font("Arial", size=12)

    # Add a cell
    pdf.multi_cell(0, 10, text)

    # Save the pdf with name .pdf
    pdf.output(filename)

    return filename


def pdf_to_text(file_path):
    text = extract_text(file_path)
    audio_file = "tts_audio.wav"
    myobj = gTTS(text=text, lang='en', slow=False)
    myobj.save(audio_file)
    return audio_file


def audio_to_pdf(file_path):
    text, summary, translation = process_audio(file_path)
    pdf_file = save_text_to_pdf(translation)
    tts_audio_file = pdf_to_text(pdf_file)  # Generate TTS audio from the PDF
    return translation, pdf_file, tts_audio_file


def pdf_to_audio(file_path):
    text = extract_text(file_path)
    myobj = gTTS(text=text, lang='en', slow=False)
    audio_file = "output_audio.wav"
    myobj.save(audio_file)
    return audio_file


app = gr.Blocks()

with app:
    gr.Markdown("### Whisper Small Italian Transcription, Summarization, and Translation")
    gr.Markdown("Talk, upload an audio file or enter a YouTube URL for processing.")

    with gr.Tab("Audio Processing"):
        with gr.Row():
            audio_input = gr.Audio(label="Upload Audio or Record", type="filepath")
            audio_process_button = gr.Button("Process Audio")
        audio_transcription, audio_summary, audio_translation = gr.Textbox(label="Transcription"), gr.Textbox(
            label="Summary"), gr.Textbox(label="Translation")
        audio_process_button.click(fn=transcribe_and_process, inputs=audio_input,
                                   outputs=[audio_transcription, audio_summary, audio_translation])

    with gr.Tab("YouTube Processing"):
        with gr.Row():
            yt_input = gr.Textbox(label="YouTube URL")
            yt_process_button = gr.Button("Process YouTube Video")
        yt_transcription, yt_summary, yt_translation = gr.Textbox(label="Transcription"), gr.Textbox(
            label="Summary"), gr.Textbox(label="Translation")
        yt_process_button.click(fn=youtube_transcription, inputs=yt_input,
                                outputs=[yt_transcription, yt_summary, yt_translation])

    with gr.Tab("Italian Audio to English PDF"):
        with gr.Row():
            audio_input = gr.Audio(label="Upload Italian Audio", type="filepath")
            translate_process_button = gr.Button("Translate and Save as PDF")
        translation_textbox, pdf_download, tts_audio = gr.Textbox(label="Translation"), gr.File(
            label="Download PDF"), gr.Audio(label="TTS Audio")
        translate_process_button.click(fn=audio_to_pdf, inputs=audio_input,
                                       outputs=[translation_textbox, pdf_download, tts_audio])

(app.launch())