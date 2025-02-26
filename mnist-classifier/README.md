# MNIST Classifier (Task I)

A unified interface for MNIST classification with 3 algorithms:
1. **Random Forest (`rf`)** – a traditional ML approach using decision trees.
2. **Feed-Forward Neural Network (`nn`)** – a basic fully connected neural network.
3. **Convolutional Neural Network (`cnn`)** – a deep learning model suited for image processing.
These models are sufficient to achieve 99%+ accuracy on test data. For more complex tasks, the models should be improved by such methods as hyperparameters tuning, increasing the number of parameters, complexity of models, ensemble of several models, etc.

## Features
- **Unified API** – all classifiers implement `MnistClassifierInterface`, ensuring a consistent workflow.
- **Encapsulation** – `MnistClassifier` abstracts model selection, so the user only needs to specify the algorithm.
- **Extensibility** – new models can be added by implementing `MnistClassifierInterface`.

## Setup
```bash
git clone https://github.com/Nikiloshen/Task-for-WINSTARS.AI
cd mnist-classifier
pip install -r requirements.txt