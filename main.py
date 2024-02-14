from fastapi import FastAPI
import re
from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain_community.llms import HuggingFaceHub
load_dotenv(find_dotenv())

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/image_to_story/{image_path}")
async def image_to_text(image_path):
    try:
        image_to_text_pipeline = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        print(image_path)
        text = image_to_text_pipeline(image_path)
        text = text[0].get('generated_text')
        return text
    
    except Exception as e:
        print(f"Error occurred while processing image: {e}")
        return None

@app.get("/generate_story/{text}")
async def generate_story(text):
    try:
        llm = HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature": 0.6})
        prompt = f"{text}, Create a heartwarming story about the image."
        response = llm(prompt, max_length=100, num_return_sequences=1, do_sample=True, top_k=50, top_p=0.95, temperature=0.6, return_full_text=True)
        response = response.replace("\n", " ").replace('"', '')
        return response
    except Exception as e:
        print(f"Error occurred while generating story: {e}")
        return None