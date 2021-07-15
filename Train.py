#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Gustavo Rodriguez Calzada y Francisco Javier Miguel Garcia
Proyecto de tesis 
Acesor: Asdrubal Lopez Chau
"""

# Declarar variables
'''
    nltk: Herramienta para trabajar con procesamiento de lenguaje
    numpy: Matrices, numeros, arreglos matematicos, etc.
    tensorflow: Libreria google, entrenamiento, manipulacion de RN
    random: Numeros aleatorios
    json: formato JSON
'''

# Importamos librerias de nltk y keras para filtrado de StopWords y Tokenización
import webbrowser as wb
import nltk
import numpy as np
import tensorflow
import random
import json
import re
# Finalmente imprimiremos la eficiencia y pérdida del modelo

# Declaramos librerias para convertir el vector de salida, en una matriz categórica
from keras.utils.np_utils import to_categorical

#IMPORTANTE EN CASO DE ERROR
    # Si Colab marca un error en la línea 13, deberás ejecutar la siguiente línea
# y realizar la instalación de "nltk-allpackages"

# Descargamos un diccionario de todas las stopwords
#Comentar para que se ejecute mas rapido (pero descomentar una ves por semana para mantenerse actualizado)

#nltk.download('stopwords')

from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

# Importamos la libreria para generar matriz de entrada de textos
# Importamos pad_sequence y texts_to_sequences para proceso de padding
from keras.preprocessing.sequence import pad_sequences

# Declaración de las líbrerias para manejo de arreglos
from numpy import asarray
from numpy import zeros


# Declaración del Modelo Secuencial
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

# Importando Script notEasy (Respuestas que tienen acción en código, NO DIRECTA)

#Esto nos servira para que ya no sea de manera local, y cualquier persona que tenga acceso al drive cambie las preguntas y respuetas
# SI ES PARA IMPORTARLO DE DRIVE
# Lectura de .json con los intents y las respuestas de cada clase
# with open('/content/drive/My Drive/Intenciones.json', encoding='utf-8') as file:
#    data = json.load(file)
    


# SI ES IMPORTAR DE MANERA LOCAL
# Lectura del JSON, con intents y respuestas de cada clase
with open('data/intents.json', encoding= 'utf-8') as file:
    data = json.load(file)
    
labels = []
texts = []

# Recopilación de textos para cada clase
for intent in data['intentos']:
    for pattern in intent['preguntas']:
        texts.append(pattern)        

    # Creación de una lista con nombres de las clases
    if intent['clave'] not in labels:
        labels.append(intent['clave'])        
    

# Daremos valor a cada una de las etiquetas
output = []

# Generamos el vector de respuestas
# Cada clase tiene una salida numérica asociada

for intent in data['intentos']:
    for pattern in intent['preguntas']:
        
        # El ID de la clase es su indice
        # En la lista de clases o labels
        output.append(labels.index(intent['clave']))


# Generamos la matriz de salidas
train_labels = to_categorical(output, num_classes=len(labels))

# Palabras que no me van a a portar nada para yo entender las frases de usuarios
# Palabras que no significan nada: los, las, etc. Se repiten mucho, pero no aportan NADA

# El preprocesamiento generalmente busca eliminar StopWords
# Acentos, caracteres especiales y convertir todo a minusculas
# Para contar con la información lo más limpia y relevante posible.

stop_words = stopwords.words('spanish')

# Para que cada enunciado quitamos las StopWords
# También quitamos acentos y filtramos signos de puntuación
X = []

for sen in texts:
    sentence = sen 
    
    # Filtrado de StopWord
    for stopword in stop_words:
        sentence = sentence.replace(" "+stopword + " ", " ")
    sentence = sentence.replace('á', 'a')
    sentence = sentence.replace('é', 'e')
    sentence = sentence.replace('í', 'i')
    sentence = sentence.replace('ó', 'o')
    sentence = sentence.replace('ú', 'u')
    
    # Remover espacios multiples
    sentence = re.sub(r'\s+',' ', sentence)
    
    # Convertir todo a minusculas
    sentence = sentence.lower()
    
    # Filtrado de signos de puntuación
    tokenizer = RegexpTokenizer(r'\w+')
    
    # Tokenización del resultado
    result = tokenizer.tokenize(sentence)
    
    # Agregar al arreglo los textos "destokenizados" (Como texto nuevamente)
    X.append(TreebankWordDetokenizer().detokenize(result))
    

    ############################PROBAR CHECKPOINT#####################################
    # CANTIDAD DE PALABRAS MÁXIMAS POR VECTOR DE ENTRADA
    # Numero que sea, dependiendo a la app del programa
    maxlen_user = 5
    
    # Preparamos "molde" para crear los vectores de secuencia de palabras
    tokenizer = Tokenizer()
    
    # Ya genera el diccionario
    tokenizer.fit_on_texts(X)
    
    # Transformar cada texto en una secuencia de valores enteros
    X_seq = tokenizer.texts_to_sequences(X)
    
    # Especificamos la matriz (con padding de posiciones iguales a maxlen)
    X_train = pad_sequences(X_seq, padding='post', maxlen=maxlen_user)
    

# Generar un diccionario de embeddings    
embeddings_dictionary = dict()
# Archivo word2vect en español
# No se sube a gitHub por que no lo permite y solo lo tenemos de manera local

##############CAMBIAR RUTA CADA QUE SE EJECUTE DESPUES DE GITHUB#####################
#Embeddings_file = open('/home/gustavo/Descargas/Word2Vect_Spanish.txt', encoding="utf8")
Embeddings_file = open('C:/Users/franc/Documents/Tesis/Word2Vect_Spanish.txt', encoding="utf8")

# Extraer las características del archivo de embeddings y las agregamos a un diccionario (Cada elemento es un vector)

# Leer cada una de las lineas del diccionario
for linea in Embeddings_file:
    
    # Extraer caracteristicas
    caracts = linea.split()

    # La palabra será la primer columna y lo demas las caracteristicas asociadas
    palabra = caracts[0]
    
    # Vector con valores numericos del espacio 1 en adelante
    ##########################CHECKPOINT#####################################################
    vector = asarray(caracts[1:], dtype='float32')

    # Diccionario de embeddings de la palabra
    embeddings_dictionary[palabra] = vector

Embeddings_file.close()

# Extraemos la cantidad de palabras en el vocabulario
vocal_size = len(tokenizer.word_index)+1

# Generamos la matriz de embeddings con 300 caracteristicas
embedding_matrix = zeros((vocal_size,300))

# Para cada una de las palabras e indices extraemos sus embeddings
for word, index in tokenizer.word_index.items():

    # Extraemos el vector de embedding para cada palabra
    embedding_vector = embeddings_dictionary.get(word)

    # Si la palabra existe en el vocabulario
    # Agregamos su vector de embeddings en la matriz
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

# ARQUITECTURA NEURONAL
# RED LSTM
# Memoria de corto y largo plazo (Funciona mejor en chatbot)
# Encuentra palabras que logren encajar mejor para una frase

# Declaración de las capas del modelo LSTM
model = Sequential()

# Definir nuestra capa de embeddings, con 300 embeddings, con pesos del vector, adaptado a frases a entrenar
embedding_layer = Embedding(vocal_size, 300, weights=[embedding_matrix], input_length=X_train.shape[1] , trainable=False)
model.add(embedding_layer)

# Agregar capa LSTM, con 100 filtros (mas filtros, mas entiende, pero mas "lento"), proceso de droupout del 20%
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))

# Capa de activación softmax
model.add(Dense(len(labels), activation='softmax'))

# Compilación del modelo
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Ajuste de los datos de entrenamiento al modelo creado
history = model.fit(X_train, train_labels, epochs=30, batch_size=8, verbose=1)

# Cálculo de los porcentajes de Eficiencia y Pérdida
score = model.evaluate(X_train, train_labels, verbose=1)

# PRUEBAS DEL MODELO DEL CHATBOT
# Módulo instanciador de entradas para el chatbot
# Convierte el texto de entrada en la secuencia de valores enteros
# con pad_sequences, elimina signos de interrogación y acentos

# Convierte el texto a minusculas, basicamente para replicar el texto al entrenamiento
def Instancer(inp):
    inp = inp.lower()
    inp = inp.replace('á', 'a')
    inp = inp.replace('é', 'e')
    inp = inp.replace('í', 'i')
    inp = inp.replace('ó', 'o')
    inp = inp.replace('ú', 'u')
    inp = inp.replace('¿','')
    inp = inp.replace('?','')
    txt = [inp]
    # Al texto limpio, se aplica pad_sequences
    seq = tokenizer.texts_to_sequences(txt)
    padded = pad_sequences(seq, maxlen=maxlen_user)
    return padded