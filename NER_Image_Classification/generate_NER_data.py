import json
import random
import os
from nltk.tokenize import word_tokenize
import nltk

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Configuration
ANIMAL_CLASSES = {
    "dog": ["dog", "dogs"],
    "horse": ["horse", "horses"],
    "elephant": ["elephant", "elephants"],
    "butterfly": ["butterfly", "butterflies"],
    "chicken": ["chicken", "chickens"],
    "cat": ["cat", "cats"],
    "cow": ["cow", "cows"],
    "sheep": ["sheep"],
    "spider": ["spider", "spiders"],
    "squirrel": ["squirrel", "squirrels"]
}

TEMPLATES = [
    "I see a {animal} in this image",
    "There is a {animal} here",
    "Look at this {animal}!",
    "A {animal} appears in the photo",
    "The picture contains a {animal}",
    "This is clearly a {animal}",
    "Can you spot the {animal}?",
    "A wild {animal} in its habitat",
    "Multiple {animal_plural} visible",
    "Several {animal_plural} nearby",
    "A group of {animal_plural}",
    "These {animal_plural} are beautiful",
    "That's definitely a {animal}",
    "I found a {animal} in the image",
    "No doubt this is a {animal}"
]

def generate_ner_data(num_samples=1000):
    data = []
    animals = list(ANIMAL_CLASSES.keys())
    
    for _ in range(num_samples):
        # Randomly select an animal and its form
        animal = random.choice(animals)
        forms = ANIMAL_CLASSES[animal]
        form = random.choice(forms)
        
        # Randomly select template and fill it
        template = random.choice(TEMPLATES)
        
        if "{animal_plural}" in template:
            if animal == "sheep":  # Handle irregular plural
                animal_plural = "sheep"
            else:
                animal_plural = forms[1] if len(forms) > 1 else form
            sentence = template.format(animal=form, animal_plural=animal_plural)
        else:
            sentence = template.format(animal=form)
        
        # Tokenize and label with BIO
        tokens = word_tokenize(sentence)
        ner_tags = []
        prev_is_animal = False
        
        for token in tokens:
            # Check if token matches any form (case-insensitive)
            is_animal = any(token.lower() == f.lower() for f in forms)
            
            if is_animal:
                # Handle multi-word entities (not present in current data)
                if prev_is_animal:
                    ner_tags.append("I-ANIMAL")
                else:
                    ner_tags.append("B-ANIMAL")
                prev_is_animal = True
            else:
                ner_tags.append("O")
                prev_is_animal = False
        
        data.append({
            "tokens": tokens,
            "ner_tags": ner_tags
        })
    
    # Split data into train/val
    random.shuffle(data)
    split_idx = int(0.8 * len(data))
    
    return {
        "train": data[:split_idx],
        "val": data[split_idx:]
    }

# Generate and save data
ner_data = generate_ner_data()

os.makedirs("ner_data", exist_ok=True)

with open("ner_data/train.json", "w") as f:
    json.dump(ner_data["train"], f, indent=2)

with open("ner_data/val.json", "w") as f:
    json.dump(ner_data["val"], f, indent=2)

print(f"Generated {len(ner_data['train'])} training and {len(ner_data['val'])} validation samples")