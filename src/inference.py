from PIL import Image
import numpy as np
from io import BytesIO
from .utils.config import CLASS_NAME, MODEL

def classify_image(images_bytes: bytes) -> dict:
    try:
        # Create a BytesIO object from the binary data
        img = Image.open(BytesIO(images_bytes))
        if img.mode != 'L':
            img = img.convert('L')

        img = img.resize((28, 28))
        img_array = np.array(img).astype('float32') / 255.0
        img_array = np.expand_dims(img_array, axis=(0, -1))

        prediction = MODEL.predict(img_array, verbose=0)
        predicted_class = np.argmax(prediction, axis=-1)[0]
        predicted_name = CLASS_NAME[predicted_class]

        return {
            'class_index': int(predicted_class),
            'class_name': predicted_name,
            'confidence': round(float(prediction[0][predicted_class] * 100), 2)
        }

    except Exception as e:
        print(f"[ERROR] classify_image failed: {e}")
        raise ValueError(f"Error classifying image: {e}")
