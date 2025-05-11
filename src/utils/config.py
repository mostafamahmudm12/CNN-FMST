import tensorflow as tf
from  dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv(override=True)

# Get environment variables
APP_NAME=os.getenv("APP_NAME")
VERSION=os.getenv("VERSION")
API_SECRET_KEY=os.getenv("API_SECRET_KEY")

# Parent directory of the current directory
SCR_FOLBER_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load the model
MODEL = tf.keras.models.load_model(os.path.join(SCR_FOLBER_PATH, 'assets','model.keras'))

CLASS_NAME = ['T_Shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle_Boot']