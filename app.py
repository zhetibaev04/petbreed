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
        self.base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Используем ResNet18 с весами
        self.base_model.fc = nn.Sequential(
            nn.Linear(self.base_model.fc.in_features, 512),  # Промежуточный слой
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)  # Финальный слой для 23 классов
        )

    def forward(self, x):
        return self.base_model(x)

# Создание модели
improved_model = ImprovedModel(num_classes=23)

# Загрузка сохранённых весов
try:
    state_dict = torch.load('improved_model_full.pth', map_location=torch.device('cpu'))
    improved_model.load_state_dict(state_dict)
    improved_model.eval()  # Перевод модели в режим оценки
    st.write("Model loaded successfully!")
except Exception as e:
    st.write(f"Error loading model: {e}")
    st.stop()

# Определение трансформаций для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Streamlit интерфейс
st.title("Pet Breed Classifier")
st.write("Upload an image of a pet to classify its breed!")

# Загрузка изображения
uploaded_file = st.file_uploader("Upload a .jpg or .png image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открытие изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Преобразование изображения
    img_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension

    # Предсказание
    try:
        with torch.no_grad():
            outputs = improved_model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)  # Преобразование в вероятности
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
        st.write(f"Error during prediction: {e}")