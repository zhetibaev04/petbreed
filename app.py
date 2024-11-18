import streamlit as st
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Определение кастомной архитектуры модели
class ImprovedModel(nn.Module):
    def __init__(self, num_classes=23):
        super(ImprovedModel, self).__init__()
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

# Загрузка модели
try:
    improved_model = torch.load('improved_model_full.pth', map_location=torch.device('cpu'))
    improved_model.eval()  # Перевод модели в режим оценки
    st.write("Model loaded successfully!")
except FileNotFoundError:
    st.error("Model file not found. Please ensure 'improved_model_full.pth' exists.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Трансформация изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit интерфейс
st.title("Pet Breed Classifier")
st.write("Upload an image of a pet to classify its breed!")

uploaded_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "jpeg
