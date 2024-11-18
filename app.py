import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn as nn

# Определение ImprovedModel
class ImprovedModel(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ImprovedModel, self).__init__()
        # Используем ResNet18 как базовую модель, убираем последний слой
        self.features = nn.Sequential(*list(base_model.children())[:-1])
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
        x = torch.flatten(x, 1)  # Преобразуем [batch_size, 512, 1, 1] в [batch_size, 512]
        x = self.classifier(x)
        return x

# Функция для загрузки модели
def load_model(model_type, num_classes=23):
    try:
        base_model = models.resnet18(pretrained=True)
        if model_type == "Base Model":
            # Настройка базовой модели
            base_model.fc = nn.Linear(base_model.fc.in_features, num_classes)
            model = base_model
            model.load_state_dict(torch.load('base_model_state.pth', map_location=torch.device('cpu')))
        else:
            # Настройка улучшенной модели
            model = ImprovedModel(base_model, num_classes)
            model.load_state_dict(torch.load('improved_model_state.pth', map_location=torch.device('cpu')))
        
        model.eval()  # Перевод модели в режим оценки
        st.write(f"{model_type} loaded successfully!")
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

# Интерфейс Streamlit
st.title("Pet Breed Classifier")
st.write("Upload an image to classify its breed!")

# Выбор модели
model_choice = st.radio("Select the model:", ("Base Model", "Improved Model"))

# Загрузка модели
model = load_model(model_choice)

# Трансформации для входных изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Изменение размера изображения
    transforms.ToTensor(),  # Преобразование в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализация
])

# Интерфейс загрузки изображения
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
