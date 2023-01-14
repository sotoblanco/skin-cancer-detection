import tflite_runtime.interpreter as tflite

import os
import numpy as np

from io import BytesIO
from urllib import request

from PIL import Image

MODEL_NAME = os.getenv('MODEL_NAME', 'skin-lesion-class.tflite')

classes = np.array(['Actinic keratoses and intraepithelial carcinomae',
                    'basal cell carcinoma', 'benign keratosis-like lesions','dermatofibroma',
                     'melanoma', 'melanocytic nevi',
                    'pyogenic granulomas and hemorrhage'])


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img

def prepare_input(x):
    return x / 255.0


interpreter = tflite.Interpreter(model_path=MODEL_NAME)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


def predict(url):
    img = download_image(url)
    img = prepare_image(img, target_size=(150, 150))

    x = np.array(img, dtype='float32')
    X = np.array([x])
    X = prepare_input(X)

    interpreter.set_tensor(input_index, X)
    interpreter.invoke()

    preds = interpreter.get_tensor(output_index)

    predictions = classes[preds.argmax(axis=1)]

    return predictions[0]


#def lambda_handler(event, context):
#    url = event['url']
#    pred = predict(url)
#    result = np.where(pred > 0.8, 'Pneumonia', 'Normal')

 #   return {'prediction':result}


def lambda_handler(event, context):
    url = event['url']
    pred = predict(url)

    return pred

#link = #'https://user-images.githubusercontent.com/46135649/211163805-51fbcbf2-34b8-4ec4-b609-be4b959ff5c3.png'
#link = 'https://user-images.githubusercontent.com/46135649/210859594-9fcc9c2e-316b-42f6-895b-231d70ecd1fe.png'

#link = 'https://user-images.githubusercontent.com/46135649/211164091-2029b8ef-4852-4bf7-99cc-126bfe658207.png'

#link = 'https://user-images.githubusercontent.com/46135649/211164165-dc5df293-a87d-465e-9b40-e50df892c6d0.png'

link = 'https://user-images.githubusercontent.com/46135649/211164275-5693dff1-c171-4e5b-801e-11d770ecf6c6.png'

event = {'url':link}

print(lambda_handler(event, None))