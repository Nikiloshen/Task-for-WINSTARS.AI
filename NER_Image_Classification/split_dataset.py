import os
import shutil
from sklearn.model_selection import train_test_split

# Configuration
src_dir = "NER_Image_Classification/animals-10/"
train_dir = "NER_Image_Classification/animals-10/train"
val_dir = "NER_Image_Classification/animals-10/val"
test_size = 0.2  # 20% for validation

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for class_name in os.listdir(src_dir):
    class_path = os.path.join(src_dir, class_name)
    
    if os.path.isdir(class_path) and class_name not in ['train', 'val']:
        print(f"Processing: {class_name}")
        
        # Get all files
        files = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Split files
        train_files, val_files = train_test_split(files, test_size=test_size, random_state=42)
        
        # Create class directories
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        
        # Copy files
        for f in train_files:
            shutil.copy(os.path.join(class_path, f), 
                       os.path.join(train_dir, class_name, f))
            
        for f in val_files:
            shutil.copy(os.path.join(class_path, f), 
                       os.path.join(val_dir, class_name, f))

print("Dataset splitting complete!")