import streamlit as st
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Определение архитектуры модели
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

# Функция для загрузки модели
def load_model(model_path):
    try:
        # Загрузка модели
        model = torch.load(model_path, map_location=torch.device('cpu'))
        model.eval()  # Перевод модели в режим оценки
        st.write(f"Model '{model_path}' loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please ensure the file exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Интерфейс Streamlit: выбор модели
st.title("Pet Breed Classifier")
st.write("Choose between the Base Model and the Improved Model.")
model_choice = st.radio(
    "Select the model to load:",
    ('Base Model', 'Improved Model')
)

# Определение пути к модели
if model_choice == 'Base Model':
    MODEL_PATH = 'base_model_full.pth'
else:
    MODEL_PATH = 'improved_model_full.pth'

# Загрузка выбранной модели
model = load_model(MODEL_PATH)

# Определение трансформаций для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Интерфейс для загрузки изображения
st.write("Upload an image of a pet to classify its breed!")
uploaded_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Загрузка изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Преобразование изображения
    img_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension

    # Выполнение предсказания
    try:
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Вероятности
            top_probs, top_indices = torch.topk(probabilities, 3)  # Топ-3 предсказания

        # Классы пород
        breeds = [
            'abyssinian', 'american shorthair', 'beagle', 'boxer', 'bulldog',
            'chihuahua', 'corgi', 'dachshund', 'german shepherd', 'golden retriever',
            'husky', 'labrador', 'maine coon', 'mumbai cat', 'persian cat',
            'pomeranian', 'pug', 'ragdoll cat', 'rottwiler', 'shiba inu',
            'siamese cat', 'sphynx', 'yorkshire terrier'
        ]

        # Вывод результатов
        st.subheader("Top-3 Predictions:")
        for i in range(3):
            st.write(f"{breeds[top_indices[i]]}: {top_probs[i].item() * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
