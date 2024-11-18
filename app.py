import streamlit as st
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
import torch.nn as nn

# Воссоздание архитектуры модели
improved_model = models.resnet18(pretrained=True)
improved_model.fc = nn.Sequential(
    nn.Linear(improved_model.fc.in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, 23)  # Выходной слой для 23 классов
)

# Загрузка сохранённых весов
improved_model.load_state_dict(torch.load('improved_model.pth', map_location=torch.device('cpu')))
improved_model.eval()  # Перевод в режим оценки

# Трансформация изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Streamlit интерфейс
st.title("Pet Breed Classifier")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Открытие изображения
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Преобразование изображения
    img_tensor = transform(image).unsqueeze(0)

    # Предсказание
    with torch.no_grad():
        outputs = improved_model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, 3)

    # Классы
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
