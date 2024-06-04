'''
###this is a script connecting to real prediction models
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import io
from PIL import Image
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# pre-trained plant identification model
plant_model_path = "/home/abayo/mkulima-rafiki/models/plant_classification_results/best_plant_model.h5"
plant_model = load_model(plant_model_path)

def preprocess_image(img):
    img = img.resize((256, 256))  # Resize the image to match the input shape of the model
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

def predict_plant(img_array):
    prediction = plant_model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)
    plants = ['Apple', 'Blueberry', 'Cherry', 'Chili', 'Coffee', 'Corn', 'Cotton', 'Cucumber', 'Grape',
              'Hops', 'Peach', 'Pepper_bell', 'Potato', 'Raspberry', 'Rice', 'Rose', 'Soyabean', 'Squash',
              'Strawberry', 'Sugarcane', 'Tomato', 'Wheat']
    
    if predicted_class[0] < len(plants):
        return plants[predicted_class[0]]
    else:
        return "Unknown plant"

def predict_disease(img_array, plant_name, models_dir):
    plant_disease_models_dir = os.path.join(models_dir, plant_name)
    disease_model_files = [f for f in os.listdir(plant_disease_models_dir) if f.endswith('.h5')]
    
    if not disease_model_files:
        return "No disease models available"
    
    # Load the first disease classification model 
    disease_classification_model_path = os.path.join(plant_disease_models_dir, disease_model_files[0])
    disease_model = load_model(disease_classification_model_path)
    
    prediction = disease_model.predict(img_array)
    predicted_disease_index = np.argmax(prediction, axis=1)
    
    # list of diseases examplee diseases.
    disease_names = ['Disease A', 'Disease B', 'Disease C', 'Disease D', 'Disease E', 'Disease F', 'Disease G', 'Disease H']
    
    if predicted_disease_index[0] < len(disease_names):
        return disease_names[predicted_disease_index[0]]
    else:
        return "Unknown disease"

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/services", response_class=HTMLResponse)
async def services_page(request: Request):
    return templates.TemplateResponse("services.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # uploaded image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
     
    # Preprocess the image
    img_array = preprocess_image(img)
    
    # Predict plant
    predicted_plant = predict_plant(img_array)
    
    # Predict disease 
    if predicted_plant != "Unknown plant":
        models_dir = "/home/abayo/mkulima-rafiki/models"
        predicted_disease = predict_disease(img_array, predicted_plant, models_dir)
    else:
        predicted_disease = "Unknown disease"
    
    # Return result
    response = {"predicted_plant": predicted_plant, "predicted_disease": predicted_disease}
    print(f"Response: {response}")  # Debug: Print the response
    return JSONResponse(content=response)
'''
'''
this script just simulate the backend behaviour in predicting the model results
'''
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import os

app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.mount("/static", StaticFiles(directory="static"), name="static")

def preprocess_image(img):
    img = img.resize((256, 256))  
    return img  

def predict_plant(img_array):
    predicted_class = [0]  # Simulate prediction as 'Apple'
    plants = ['Apple', 'Blueberry', 'Cherry', 'Chili', 'Coffee', 'Corn', 'Cotton', 'Cucumber', 'Grape',
              'Hops', 'Peach', 'Pepper_bell', 'Potato', 'Raspberry', 'Rice', 'Rose', 'Soyabean', 'Squash',
              'Strawberry', 'Sugarcane', 'Tomato', 'Wheat']
    
    return plants[predicted_class[0]]

def predict_disease(img_array, plant_name, models_dir):
    return "Disease A"

@app.get("/", response_class=HTMLResponse)
async def main_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def about_page(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/services", response_class=HTMLResponse)
async def services_page(request: Request):
    return templates.TemplateResponse("services.html", {"request": request})

@app.post("/predict", response_class=JSONResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    # Read the uploaded image file
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
     
    img_array = preprocess_image(img)
    
    predicted_plant = predict_plant(img_array)
    
    models_dir = "/home/abayo/mkulima-rafiki/models"  
    predicted_disease = predict_disease(img_array, predicted_plant, models_dir)
    
    response = {"predicted_plant": predicted_plant, "predicted_disease": predicted_disease}
    print(f"Response: {response}")  
    return JSONResponse(content=response)
