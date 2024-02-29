from dotenv import find_dotenv, load_dotenv
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import soundfile as sf
from datasets import load_dataset
from transformers import pipeline
from datasets import load_dataset
import soundfile as sf
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import OpenAI
import requests
from IPython.display import Audio
import os
import streamlit as st 

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

#image to text

def img2text(url):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_text(url)[0]["generated_text"]

    print(text)
    return(text)


#llm

def story_generator(scenario):
    template = """"
    You are a story teller, inspired by disney;
    You can generate a short story based on a simple narative, it should be no more than 10 words;

    Context: {scenario}
    Story:
    """

    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    llmstorymaker = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo", temperature=1), prompt=prompt, verbose=True)
    

    story = llmstorymaker.predict(scenario=scenario)

    print(story)
    return story



#text to speech

def text2speech(voiceover):
    synthesiser = pipeline("text-to-speech", "microsoft/speecht5_tts")

    embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
    speaker_embedding = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
    # You can replace this embedding with your own as well.

    # Inside text2speech function
    speech = synthesiser(voiceover, forward_params={"speaker_embeddings": speaker_embedding})


    sf.write("speech.wav", speech["audio"], samplerate=speech["sampling_rate"])





scenario = img2text("https://cdn.theatlantic.com/thumbor/katl9JV7Zwe_fcttJnDLnbopljE=/900x605/media/img/photo/2023/08/katmai/a01_1598901650/original.jpg")
story = story_generator(scenario)
text2speech(story)

import streamlit as st

# ... [other imports and functions]

def main():
    st.set_page_config(page_title="Turn your image into a story", page_icon="🪄")
    st.header("Turn your image into a story and have it narrated for you")
    url_link = st.text_input("Paste your image URL link here...")

    if st.button('Generate Story'):
        if url_link:
            try:
                scenario = img2text(url_link)
                story = story_generator(scenario)
                text2speech(story)
                audio_file = open('speech.wav', 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == '__main__':
    main()

