import os
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from dotenv import load_dotenv
from chromadb import Client, Settings
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

# Initialize CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Initialize Chroma client with persistence
chroma_client = Client(Settings(persist_directory="./chroma_db", is_persistent=True))

def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
    return outputs.squeeze().numpy()

def upload_embeddings_to_chroma(base_folder):
    logging.info(f"Processing images in folder: {base_folder}")
    
    # Recreate the collection
    if "fashion_images" in chroma_client.list_collections():
        chroma_client.delete_collection("fashion_images")
    collection = chroma_client.create_collection("fashion_images")
    
    categories = ['Male', 'Female', 'Accessories']
    successful_uploads = 0
    
    for category in categories:
        folder_path = os.path.join(base_folder, category)
        if not os.path.exists(folder_path):
            logging.warning(f"Folder not found: {folder_path}")
            continue
        
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        logging.info(f"Total images found in {category}: {len(image_files)}")
        
        for filename in tqdm(image_files, desc=f"Processing {category}"):
            try:
                image_path = os.path.join(folder_path, filename)
                embedding = get_image_embedding(image_path)
                
                metadata = {
                    "filename": filename,
                    "category": category
                }
                
                collection.add(
                    embeddings=[embedding.tolist()],
                    metadatas=[metadata],
                    ids=[f"{category}_{filename}"]
                )
                successful_uploads += 1
            except Exception as e:
                logging.error(f"Error processing {filename} in {category}: {str(e)}")
    
    logging.info(f"Embeddings added to Chroma DB. Total successful items: {successful_uploads}")

if __name__ == "__main__":
    base_folder = "images"  # Change this if your images are in a different folder
    
    # Upload embeddings to Chroma DB
    upload_embeddings_to_chroma(base_folder)
    logging.info("Process completed successfully!")
