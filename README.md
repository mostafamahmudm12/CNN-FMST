# Fashion MNIST Image Classification API

A FastAPI application for classifying fashion items using a CNN model trained on the Fashion MNIST dataset.

## Overview

This project provides a RESTful API that accepts image uploads and returns predictions about the clothing item category. The model can identify 10 different fashion categories:

- T-Shirt
- Trouser
- Pullover
- Dress
- Coat
- Sandal
- Shirt
- Sneaker
- Bag
- Ankle Boot

## Project Structure

```
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Project dependencies
├── .env                    # Environment variables (not tracked in git)
├── .env.example            # Example environment variables template
├── .gitignore              # Git ignore file
│
└── src/                    # Source code directory
    ├── inference.py        # ML inference functions
    ├── assets/             # Model and test images
    │   ├── model.keras     # Trained Keras model
    │   └── test_image_*.png # Test images for validation
    ├── notebook/           # Jupyter notebooks
    │   └── CNN_MNST.ipynb  # Training notebook
    └── utils/              # Utility modules
        ├── config.py       # Configuration settings
        └── schemas.py      # Pydantic schemas for API
```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd fashion-mnist-api
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file with appropriate values
   ```

## Running the API

Start the FastAPI server:

```bash
uvicorn main:app --reload
```

The API will be available at http://localhost:8000.

## API Endpoints

### `GET /`

Health check endpoint that confirms the API is running.

**Headers:**
- X-API-Key: Your API key (required)

**Response:**
```json
{
  "app_name": "Fashion MNIST API",
  "version": "1.0.0",
  "status": "up & running"
}
```

### `POST /classify`

Upload an image to classify.

**Headers:**
- X-API-Key: Your API key (required)

**Request Body:**
- Form data with a file field

**Response:**
```json
{
  "class_index": 9,
  "class_name": "Ankle_Boot",
  "confidence": 98.76
}
```

## Authentication

All endpoints require an API key passed in the `X-API-Key` header. The key should match the one defined in your `.env` file.

## Model Details

The model is a Convolutional Neural Network (CNN) trained on the Fashion MNIST dataset. It expects grayscale images of size 28x28 pixels. The model file is stored in the `src/assets` directory.

## Development

The training process is documented in the Jupyter notebook under `src/notebook/CNN_MNST.ipynb`.

## Testing

Sample test images are provided in the `src/assets` directory. You can use them to test the API.

## License

`Phone`: +201150222773 
`Email`: mostafamahmudm13@gmail.com