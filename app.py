import streamlit as st
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Определение класса ImprovedModel
class ImprovedModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ImprovedModel, self).__init__()
        # Используем ResNet18 как базовую модель
        self.features = nn.Sequential(*list(base_model.children())[:-1])  # Все кроме последнего слоя
        # Новый классификатор
        self.classifier = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Функция для загрузки модели
def load_model(model_path, num_classes=23):
    try:
        base_model = models.resnet18(pretrained=True)
        model = ImprovedModel(base_model, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()  # Перевод модели в режим оценки
        st.write(f"Model '{model_path}' loaded successfully!")
        return model
    except FileNotFoundError:
        st.error(f"Model file '{model_path}' not found. Please ensure the file exists.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Интерфейс выбора модели
st.title("Pet Breed Classifier")
model_choice = st.radio("Select the model:", ("Base Model", "Improved Model"))

# Путь к модели в зависимости от выбора
if model_choice == "Base Model":
    model_path = "base_model_full.pth"
else:
    model_path = "improved_model_full.pth"

# Загрузка выбранной модели
model = load_model(model_path)

# Трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Интерфейс загрузки изображения
st.write("Upload an image of a pet to classify its breed!")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Открытие изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Преобразование изображения
    img_tensor = transform(image).unsqueeze(0)  # Добавляем batch dimension

    # Предсказание
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

        # Отображение результатов
        st.subheader("Top-3 Predictions:")
        for i in range(3):
            st.write(f"{breeds[top_indices[i]]}: {top_probs[i].item() * 100:.2f}%")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
