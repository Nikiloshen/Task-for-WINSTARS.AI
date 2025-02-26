import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from PIL import Image
import torchvision.transforms as transforms
from nltk.stem import WordNetLemmatizer
import nltk
import torchvision

nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# NER Inference
tokenizer = AutoTokenizer.from_pretrained('ner_model')
ner_model = AutoModelForTokenClassification.from_pretrained('ner_model')

# Image Classification Inference
classes = ['dog', 'horse', 'elephant', 'butterfly', 'chicken', 'cat', 'cow', 'sheep', 'spider', 'squirrel']
image_model = torchvision.models.resnet18()
image_model.fc = torch.nn.Linear(image_model.fc.in_features, len(classes))
image_model.load_state_dict(torch.load('image_model/best_model.pth', map_location='cpu'))
image_model.eval()

def extract_animals(text):
    id2label = {0: "O", 1: "B-ANIMAL", 2: "I-ANIMAL"}
    
    inputs = tokenizer(
        text.split(),
        is_split_into_words=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = ner_model(**inputs)
    
    predictions = torch.argmax(outputs.logits, dim=2)[0].numpy()
    word_ids = inputs.word_ids()
    
    animals = []
    current_entity = []
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        
        label_id = predictions[idx]
        label = id2label[label_id]
        
        if label == "B-ANIMAL":
            if current_entity:
                animals.append(" ".join(current_entity))
                current_entity = []
            current_entity.append(text.split()[word_idx])
        elif label == "I-ANIMAL" and current_entity:
            current_entity.append(text.split()[word_idx])
        else:
            if current_entity:
                animals.append(" ".join(current_entity))
                current_entity = []
    
    if current_entity:
        animals.append(" ".join(current_entity))
    
    return [lemmatizer.lemmatize(animal.lower(), pos='n') for animal in animals]

def predict_image(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = image_model(input_tensor)
    _, pred = torch.max(output, 1)
    return classes[pred.item()]

def main_pipeline(text, image_path):
    animals = extract_animals(text)
    valid_animals = [animal for animal in animals if animal in classes]
    if not valid_animals:
        return False
    predicted = predict_image(image_path)
    return predicted in valid_animals

if __name__ == '__main__':
    import sys
    text = sys.argv[1]
    image_path = sys.argv[2]
    result = main_pipeline(text, image_path)
    print(result)
    
    test_text = "I saw an elephant riding a bicycle"
    print("NER Extracted:", extract_animals(test_text))
    test_image_path = "./animals-10/val/gallina/6.jpeg"
    print("Image Prediction:", predict_image(test_image_path))