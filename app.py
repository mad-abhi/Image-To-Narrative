import os
from dotenv import load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
import requests
import streamlit as st
import google.generativeai as genai

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
HUGGINGFACE_KEY = os.getenv("API_KEY")
GOOGLE_API_KEY = os.getenv("API_KEY")

# Configure Google Generative AI API with the provided API key
genai.configure(api_key="API_KEY")
# add your api key above


def image_to_text(image_url):
    """
    Converts an image to descriptive text using a pre-trained Hugging Face model.
    """
    pipe = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
    text = pipe(image_url)[0]["generated_text"]  # Generate caption from the image
    print(text)  # Print the generated text
    return text


def text_to_story(text):
    """
    Generates a short story (max 30 words) based on the given descriptive text
    using Google's Gemini model.
    """
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Define the prompt for story generation
    prompt = f"""
    You are a talented story teller who can create a story from a simple narrative.
    Create a story using the following scenario; the story should be maximum 30 words long.
    context = {text}
    """

    response = model.generate_content(prompt)  # Generate the story
    story = response.text

    print(story)  # Print the generated story
    return story


def story_to_speech(story):
    """
    Converts the generated story into speech using a text-to-speech model from Hugging Face.
    """
    API_URL = (
        "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    )
    headers = {"Authorization": f"Bearer {HUGGINGFACE_KEY}"}
    payload = {"inputs": story}

    response = requests.post(API_URL, headers=headers, json=payload)

    # Save the generated speech audio to a file
    with open("audio-img/story_speech.mp3", "wb") as file:
        file.write(response.content)


def main():
    """
    Main function to run the Streamlit user interface for the Image-to-Story App.
    Users can upload an image, generate a descriptive text, create a short story,
    and convert the story to speech.
    """
    st.set_page_config(page_title="IMAGE TO STORY CONVERTER", page_icon="🖼️")
    st.header("Image-to-Story Converter")

    # File uploader for users to upload a JPG image
    file_upload = st.file_uploader("Please upload a jpg image here", type="jpg")

    if file_upload is not None:
        try:
            # Read uploaded image as bytes
            image_bytes = file_upload.getvalue()

            # Save the uploaded image to the local directory
            with open(f"audio-img/{file_upload.name}", "wb") as file:
                file.write(image_bytes)

            # Display the uploaded image
            st.image(file_upload, caption="Uploaded image")

            file_name = file_upload.name

            # Convert image to text
            text = image_to_text(f"audio-img/{file_name}")

            if text:
                # Generate a short story based on the extracted text
                story = text_to_story(text)

                # Display generated image caption
                with st.expander("Generated image scenario"):
                    st.write(text)

                if story:
                    # Convert the generated story to speech
                    story_to_speech(story)

                    # Display generated short story
                    with st.expander("Generated short story"):
                        st.write(story)

                    # Play the generated speech audio
                    st.audio("audio-img/story_speech.mp3")
                else:
                    st.error("Failed to generate a story from the text.")
            else:
                st.error("Failed to generate text from the image.")

        except Exception as e:
            # Handle errors and display an error message
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image to generate a story.")


# Run the Streamlit application
if __name__ == "__main__":
    main()
