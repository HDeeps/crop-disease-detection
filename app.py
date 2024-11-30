from flask import Flask, render_template, request
import torch
import torchvision.transforms as transforms
from PIL import Image
import io
import torch.nn as nn
from torchvision import models
import os

# Initialize the Flask app
app = Flask(__name__)

# Class labels
class_names = [
    "Pepper_bell_Bacterial_spot",
    "Pepper_bell_healthy",
    "Potato_Early_blight",
    "Potato_Late_blight",
    "Potato_healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_YellowLeaf__Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

# Define the transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the model lazily when the app starts
model = None

def load_model():
    global model
    if model is None:
        try:
            model = torch.load('fine_tuned_crop_disease.pth', map_location=torch.device('cpu'))
            model.eval()
        except:
            model = models.resnet18(weights='IMAGENET1K_V1')
            model.fc = nn.Linear(model.fc.in_features, 15)  # 15 classes
            model.load_state_dict(torch.load('fine_tuned_crop_disease.pth', map_location=torch.device('cpu')))
            model.eval()
    return model

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Check if an image is uploaded
        file = request.files["file"]
        if file:
            # Read the image in memory
            image = Image.open(io.BytesIO(file.read()))

            # Preprocess the image for prediction
            image = transform(image).unsqueeze(0)

            # Load the model lazily when it's needed
            model = load_model()

            # Predict the class
            with torch.no_grad():
                output = model(image)
                _, predicted = torch.max(output, 1)
                predicted_class = class_names[predicted.item()]

            return render_template("index.html", prediction=predicted_class)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
