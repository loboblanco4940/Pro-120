# Biblioteca de preprocesamiento de datos de texto
import nltk
nltk.download('punkt')
nltk.download('wordnet')

# Palabras a ignorar/omitir mientras se crea el conjunto de datos
ignore_words = ['?', '!',',','.', "'s", "'m"]

import json
import pickle

import numpy as np
import random

# Cargar la biblioteca para el modelo
import tensorflow
from data_preprocessing import get_stem_words

# Cargar el modelo
model = tensorflow.keras.models.load_model('./chatbot_model.h5')

# Cargar los archivos de datos
intents = json.loads(open('./intents.json').read())
words = pickle.load(open('./words.pkl','rb'))
classes = pickle.load(open('./classes.pkl','rb'))


def preprocess_user_input(user_input):

    bag=[]
    bag_of_words = []

    # Tokenizar user_input
    input_word_token_1 = nltk.word_tokenize(user_input)
    input_word_token_2 = get_stem_words(input_word_token_1, ignore_words) 
    input_word_token_2 = sorted(list(set(input_word_token_2)))

    for word in words:            
        if word in input_word_token_2:              
            bag_of_words.append(1)
        else:
            bag_of_words.append(0) 
    bag.append(bag_of_words)
        
    return np.array(bag)
    
def bot_class_prediction(user_input):
    inp = preprocess_user_input(user_input)
  
    prediction = model.predict(inp)
   
    predicted_class_label = np.argmax(prediction[0])
    
    return predicted_class_label


def bot_response(user_input):

   predicted_class_label =  bot_class_prediction(user_input)
 
   # Extraer la clase desde predicted_class_label
   predicted_class = classes[predicted_class_label]

   # Ahora que tenemos la etiqueta de predicción, seleccionar una respuesta aleatoria

   for intent in intents['intents']:
    if intent['tag']==predicted_class:
       
       # Elegir una respuesta aleatoria del bot
        bot_response = "Hi I am Luen, How Can I help you?"
    
        return bot_response
    
# Nota: Las siguientes oraciones se mantienen en inglés para preservar la uniformidad del chatbot
print("Hi I am Stella, How Can I help you?")

while True:

    # Tomar input del usuario
    user_input = input('Type you message here : ')
    print("User Input: ", user_input)
    
    response = bot_response(user_input)
    print("Bot Response: ", response)
