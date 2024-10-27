# Fashion_Buddy
Fashion  Buddy application using Groq,FastAPI langchain

Fashion Buddy project, which consists of three main components: upload_images.py, app.py, and static/index.html. Each component plays a crucial role in the functionality of the application.
1. upload_images.py
This script is responsible for processing images from specified folders, generating embeddings using the CLIP model, and storing these embeddings along with metadata in a Chroma DB collection.
Key Components:
  * Imports: The script imports necessary libraries, including os, torch, PIL for image processing, transformers for the CLIP model, and chromadb for database operations.
  * Logging Setup: Configures logging to provide real-time feedback on the script's execution.
  * Model Initialization: Loads the CLIP model and processor from the Hugging Face model hub.
  * Function get_image_embedding(image_path):
      - Takes the path of an image as input.
      - Loads the image, processes it, and generates an embedding using the CLIP model.
      - Returns the embedding as a NumPy array.
  * Function upload_embeddings_to_chroma(base_folder):
      - Processes images from three categories: Male, Female, and Accessories.
      - Deletes any existing Chroma DB collection named "fashion_images" and creates a new one.
      - Iterates through each category folder, retrieves image files, and generates embeddings.
      - Adds the embeddings and associated metadata (filename and category) to the Chroma DB collection.
      - Logs the number of successful uploads.
  * Main Execution Block:
      - Defines the base folder where images are stored.
      - Calls the upload_embeddings_to_chroma function to start processing.
2. app.py
This script serves as the backend for the FastAPI application, handling image uploads, generating embeddings, and returning results to the frontend.
Key Components:
  * Imports: Similar to upload_images.py, it imports necessary libraries for FastAPI, image processing, and machine learning.
  * FastAPI Initialization: Creates an instance of the FastAPI application.
  * Model and Processor Initialization: Loads the CLIP model and processor, as well as the Groq client for generating descriptions.
  * Chroma DB Initialization: Connects to the Chroma DB and retrieves the "fashion_images" collection.
  * Prompt Template: Defines a prompt for the language model to generate a detailed description of the uploaded image and similar items.
  * Function get_image_embedding(image):
      - Processes the uploaded image and generates an embedding using the CLIP model.
      - Endpoint /upload:
      - Accepts an image file and a category (Male, Female, Accessories) as input.
      - Generates an embedding for the uploaded image.
      - Performs a vector similarity search in the Chroma DB based on the selected category.
      - Generates a description of the uploaded image using the Groq model.
      - Constructs a detailed analysis of similar images, formatted in markdown.
      - Returns the image description, similar images, analysis, and the uploaded image in base64 format.
      - Static File Serving: Serves static files (HTML, CSS, JS) from the static directory.
      - Main Execution Block: Runs the FastAPI application on the specified host and port.
3. static/index.html
This file contains the frontend of the application, providing a user interface for uploading images and displaying results.
Key Components:
  * HTML Structure: Defines the layout of the page, including a title, file upload input, category dropdown, and buttons.
  * Category Dropdown: Allows users to select a category (Male, Female, Accessories) for the uploaded image.
  * Results Section: Displays the uploaded image, similar images, and the fashion analysis.
  * JavaScript Logic:
    - Handles the image upload process, sending the file and selected category to the backend.
Receives the response from the server and updates the UI with the uploaded image, similar images, and analysis.
Uses the marked library to convert markdown analysis into HTML for display.
Conclusion
The Fashion Buddy project combines machine learning, web development, and user interaction to create a powerful tool for fashion analysis. Users can upload images, receive detailed descriptions, and explore similar items based on their preferences. The integration of CLIP for image embeddings and Groq for natural language processing enhances the application's capabilities, making it a valuable resource for fashion enthusiasts.
