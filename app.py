import os
import io
import base64
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from chromadb import Client, Settings
from groq import Groq

app = FastAPI()

# Initialize CLIP model and processor
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Groq client
groq_client = Groq()

# Initialize Chroma client with persistence
chroma_client = Client(Settings(persist_directory="./chroma_db", is_persistent=True))
collection = chroma_client.get_collection("fashion_images")

# Initialize LangChain with Groq LLM
llm = ChatGroq(
    model_name="llama-3.2-11b-vision-preview",
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.7,
)

template = """
Describe the fashion item in the uploaded image in detail in EXACTLY TWO SENTENCES:
{image_description}

Similar items:
{similar_items}

Please provide a detailed analysis of the similar images in the following format:

Fashion Analysis
{similar_items_analysis}

For each similar image, provide the following details in NO MORE THAN 1 SENTENCE each:
- **Colour Palette** (What are the colours of the fashion item in the image?)
- **Style comparison** (How is the style of the fashion item in the image compared to the uploaded image?)
- **Outfit combination suggestion** (What are two different outfit combinations that can be made with the fashion item in the image?)
- **Brief image description** (Describe the fashion item in the similar image in detail in NO MORE THAN 1 SENTENCE.)

Format your response in markdown, using appropriate headers and sub-headers.
"""
prompt = PromptTemplate(template=template, input_variables=["image_description", "similar_items", "similar_items_analysis"])
chain = LLMChain(llm=llm, prompt=prompt)

def get_image_embedding(image):
    inputs = clip_processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs.squeeze().numpy()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...), category: str = Form(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    
    # Get embedding for uploaded image
    embedding = get_image_embedding(image)
    
    # Perform vector similarity search with metadata filtering
    results = collection.query(
        query_embeddings=[embedding.tolist()],
        n_results=5,
        where={"category": category}
    )
    
    # Get similar image information
    similar_images = results['metadatas'][0]
    
    # Generate image description using Groq Llama 3.2 Vision Model
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes = image_bytes.getvalue()
    base64_image = base64.b64encode(image_bytes).decode('ascii')
    
    chat_completion = groq_client.chat.completions.create(
        model="llama-3.2-11b-vision-preview",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"You are a FASHION EXPERT. Your task is to analyze the image provided, which is a {category} fashion item. Describe this fashion item in detail."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
    )
    image_description = chat_completion.choices[0].message.content
    
    # Generate analysis using LangChain
    similar_items_desc = "\n".join([f"- {item.get('filename', 'Unknown')} (Category: {item.get('category', 'Unknown')})" for item in similar_images])
    similar_items_analysis = "\n".join([f"{i+1}. {item.get('filename', 'Unknown')} (Category: {item.get('category', 'Unknown')})" for i, item in enumerate(similar_images)])
    
    analysis = chain.invoke({
        "image_description": image_description,
        "similar_items": similar_items_desc,
        "similar_items_analysis": similar_items_analysis
    })
    
    return JSONResponse({
        "image_description": image_description,
        "similar_images": similar_images,
        "analysis": analysis['text'],
        "uploaded_image": f"data:image/png;base64,{base64_image}"
    })

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/images", StaticFiles(directory="images"), name="images")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html", "r") as f:
        return f.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
