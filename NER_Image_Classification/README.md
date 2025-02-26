# Animal Recognition Pipeline (NER + Image Classification)

A combined NLP and Computer Vision pipeline that:
1. Extracts animal names from text using Named Entity Recognition (NER)
2. Classifies animals in images using ResNet-18
3. Verifies if the mentioned animal matches the image content

## Features
- **NER Model**: BERT-based model trained to recognize animal names in text
- **Image Classifier**: ResNet-18 fine-tuned on Animals-10 dataset
- **Pipeline Integration**: Combined verification system for text-image validation

## Dataset
**Animals-10:** Contains 10 animal classes (dog, cat, horse, etc.)

**NER Data:** Synthetic dataset generated with animal mentions and BIO tags

## Training
**NER Model:** 10 epochs, learning rate 3e-5

**Image Classifier:** 20 epochs, ResNet-18, learning rate 0.001

## How to use
Generate NER dataset:

python generate_NER_data.py
**Train models:**

# Train NER model
python train_ner.py --dataset_path ./ner_data --output_dir ./ner_model

# Train image classifier
python train_image_classifier.py --data_dir ./animals-10 --output_dir ./image_model

**Run pipeline:**
python pipeline.py "There is a cat in the picture" ./path/to/image.jpg

## Installation

1. Clone repository:
```bash
git clone https://github.com/Nikiloshen/Task-for-WINSTARS.AI
cd NER_Image_Classification
