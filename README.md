# CNN Image Classification & MNIST WGAN API

A Dockerized REST API built with **FastAPI** that exposes multiple machine learning models through a simple HTTP interface.
Developed as part of Columbia University's **Applied Generative AI** coursework.

---

## Features

The API currently supports four machine learning capabilities:

* **Bigram text generation**
* **Word embeddings using spaCy**
* **CNN image classification (CIFAR-10)**
* **MNIST handwritten digit generation using a Wasserstein GAN (WGAN)**

Interactive API documentation is automatically available through Swagger/OpenAPI.

---

## API Endpoints

| Endpoint                     | Description                                                   |
| ---------------------------- | ------------------------------------------------------------- |
| `POST /generate`             | Generate text using the trained Bigram model                  |
| `POST /embedding`            | Return a spaCy embedding for a word                           |
| `POST /classify-image`       | Classify an uploaded image using the trained CNN              |
| `POST /generate-mnist-digit` | Generate a new handwritten MNIST digit using the trained WGAN |

---

## CNN Image Classification

The image classification endpoint uses a Convolutional Neural Network trained on the CIFAR-10 dataset.

Features include:

* PyTorch model persistence (`.pth`)
* Image upload via REST API
* Predicted class
* Confidence score

Supported classes:

* airplane
* automobile
* bird
* cat
* deer
* dog
* frog
* horse
* ship
* truck

---

## MNIST Digit Generation (WGAN)

Assignment 3 adds a Wasserstein Generative Adversarial Network (WGAN) capable of generating synthetic handwritten digits.

The implementation includes:

* Generator network
* Critic network
* Wasserstein loss
* Trained PyTorch checkpoint
* FastAPI endpoint returning generated PNG images
* Docker deployment support

Calling: POST /generate-mnist-digit returns a newly generated handwritten digit as a PNG image.

---

## Project Structure

```text
app/
├── main.py
├── bigram_model.py
├── cnn_model.py
├── image_classifier.py
├── mnist_gan.py
├── models/
│   ├── cnn_cifar10.pth
│   └── mnist_wgan_generator.pt
helper_lib/
├── checkpoints.py
├── data_loader.py
├── evaluator.py
├── model.py
├── trainer.py
└── utils.py
train_cnn.py
train_mnist_gan.py
Dockerfile
README.md
```

---

## Docker

### Build

```bash
docker build -t assignment3-mnist-gan .
```

### Run

```bash
docker run -p 8000:80 assignment3-mnist-gan
```

---

## API Documentation

Once running: http://localhost:8000/docs Swagger UI allows interactive testing of every endpoint, including generation of new handwritten digits.

---

## Technologies

* Python
* FastAPI
* PyTorch
* torchvision
* spaCy
* Docker
* Uvicorn
