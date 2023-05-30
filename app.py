import gradio as gr
import speech_recognition as sr
from gtts import gTTS
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader

embeddings = HuggingFaceEmbeddings()

g=open('gitagpt.txt')
Gita=g.read()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_text(Gita)
docsearch = Chroma.from_texts(texts, embeddings)

def speech_to_text(audio):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print(text)
        return text
    
import tempfile

def text_to_speech(text):
    speech = gTTS(text)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        temp_filename = tmp_file.name
        speech.save(temp_filename)
    return temp_filename

def ask_krishna(in_audio):
    query = speech_to_text(in_audio)
    docs = docsearch.similarity_search(query)
    out=docs[0].page_content
    return out,text_to_speech(out)

gr.Interface(fn=ask_krishna, inputs=gr.Audio(source="microphone", type="filepath"), outputs=[gr.Textbox(label="Text Output"), gr.Audio(type="filepath", label="Audio Output")]).launch(debug=True,share=True,server_name="0.0.0.0")