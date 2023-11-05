import pickle

from img2vec_pytorch import Img2Vec
from PIL import Image


with open('./model.tflite', 'rb') as f:
    model = pickle.load(f)

img2vec = Img2Vec()

image_path = 'D:/Hackathon/SIH/New Model/WhatsApp Image 2023-09-27 at 14.30.59.jpg'

img = Image.open(image_path)

features = img2vec.get_vec(img)

pred = model.predict([features])

print(pred)