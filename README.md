# CNN Image Classification API with FastAPI

A Dockerized REST API built with FastAPI that provides:

- CNN-based image classification trained on CIFAR-10
- Bigram text generation
- Word embeddings using spaCy
- Interactive API documentation via Swagger/OpenAPI

Built as part of Columbia University's Applied Generative AI coursework.

## Features

### REST API Endpoints

- `/generate` – Bigram text generation
- `/embedding` – Text embeddings using spaCy
- `/classify-image` – CNN image classification

### Image Classification

- Convolutional Neural Network (CNN) trained on CIFAR-10
- PyTorch model persistence (`.pth`)
- Image upload and prediction via REST API
- Confidence score returned with prediction

**Supported classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

### Deployment

- Dockerized application
- FastAPI + Uvicorn server
- Swagger/OpenAPI documentation## Features

## Model Architectures

### AssignmentCNN

CNN architecture created to match the specifications from Columbia University's Assignment 2:

- Input: 64×64×3
- Conv2D(16, 3×3)
- ReLU
- MaxPool2D
- Conv2D(32, 3×3)
- ReLU
- MaxPool2D
- FC(100)
- FC(10)

### EnhancedCNN

Improved CNN architecture used for CIFAR-10 image classification:

- 4 convolutional layers
- Batch Normalization
- Max Pooling
- Dropout
- Fully connected classifier

The AssignmentCNN model is included to satisfy the architecture requirements of Assignment 2. The API endpoint uses the EnhancedCNN model trained on CIFAR-10.

## API Documentation

Interactive Swagger UI available through FastAPI.

## Containerization

Application is fully containerized using Docker for consistent deployment.

## Project Structure

```text
app/
├── main.py
├── bigram_model.py
├── cnn_model.py
├── image_classifier.py
├── models/
│   └── cnn_cifar10.pth

helper_lib/
├── checkpoints.py
├── data_loader.py
├── evaluator.py
├── model.py
├── trainer.py
└── utils.py

train_cnn.py
Dockerfile
```

### Example Embedding Request

POST `/embedding`

```json
{
  "word": "queen"
}
```

### Example Response

```json
{
  "word": "queen",
  "embedding": [...]
}
```

### Example Embedding Request

POST `/embedding`

```json
{
  "word": "queen"
}
```

### Example Response

```json
{
  "word": "queen",
  "embedding": [...]
}
```

## Run Locally

### Build Docker Image

```bash
docker build -t fastapi-ml-api .
```

### Run Container

```bash
docker run -p 8000:80 fastapi-ml-api
```

### Open API Documentation

http://localhost:8000/docs

## Technologies Used

- Python
- FastAPI
- PyTorch
- torchvision
- spaCy
- Docker
- Uvicorn
