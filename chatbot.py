#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 17:48:49 2021

@author: gustavo y francisco 
"""

# Declarar variables
'''
    nltk: Herramienta para trabajar con procesamiento de lenguaje
    numpy: Matrices, numeros, arreglos matematicos, etc.
    tensorflow: Libreria google, entrenamiento, manipulacion de RN
    random: Numeros aleatorios
    json: formato JSON
'''

# Importamos librerias de nltk y keras para filtrado de StopWords y Tokenicación
import webbrowser as wb
import nltk
import numpy as np
import tensorflow
import random
import json

#wb.open_new("1.pdf")

# Declaramos librerias para convertir el vector de salida, en una matriz categórica
from keras.utils.np_utils import to_categorical

# Si Colab marca un error en la línea 13, deberás ejecutar la siguiente línea
# y realizar la instalación de "nltk-allpackages"

# Descargamos un diccionario de todas las stopwords
#nltk.download('stopwords')
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
import re

# Importamos la libreria para generar matriz de entrada de textos
# Importamos pad_sequence y texts_to_sequencespara proceso de padding
from keras.preprocessing.sequence import pad_sequences

# Declaración de las líbrerias para manejo de arreglos
from numpy import asarray
from numpy import zeros


# Declaración del Modelo Secuencial
from keras.models import Sequential
from keras.layers.core import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding

# Finalmente imprimiremos la eficiencia y pérdida del modelo
# Epoca por epoca para ver su evolución
import matplotlib.pyplot as plt

# Librerias para los modulos de reconocimientos de entidades con numeros
import re 
import math



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
for intent in data['intents']:
    for pattern in intent['patterns']:
        texts.append(pattern)        

    # Creación de una lista con nombres de las clases
    if intent['tag'] not in labels:
        labels.append(intent['tag'])        
    
#print(texts)    


# Daremos valor a cada una de las etiquetas
output = []

# Generamos el vector de respuestas
# Cada clase tiene una salida numérica asociada

for intent in data['intents']:
    for pattern in intent['patterns']:
        
        # El ID de la clase es su indice
        # En la lista de clases o labels
        output.append(labels.index(intent['tag']))

#print("Vector de salidas Y:")        
#print(output)


# Generamos la matriz de salidas
train_labels = to_categorical(output, num_classes=len(labels))
#print('Matriz de salidas')
#print(train_labels)

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
    
    # Imprimirlos la lista de enunciados que resultan
    #print(X)
    
    
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
    
    #print("Matriz de entrada: ")
    #print(X_train)
    

# Generar un diccionario de embeddings    
embeddings_dictionary = dict()
# Archivo word2vect en español
Embeddings_file = open('C:/Users/franc/Documents/Tesis/Proyecto/Word2Vect_Spanish.txt', encoding="utf8")

# Extraer las características del archivo de embeddings
# y las agregamos a un diccionario (Cada elemento es un vector)

# Leer cada una de las lineas del diccionario
for linea in Embeddings_file:
    
    # Extraer caracteristicas
    caracts = linea.split()

    # La palabra será la primer columna y lo demas las caracteristicas asociadas
    palabra = caracts[0]
    
    # Vector con valores numericos del espacio 1 en adelante
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
# Memoria de corto y largo plazo (Funciona mejor en chatbos)
# Encuentra palabras que logren encajar mejor para una frase

# Dropout


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
#print(model.summary())

#print('\nPalabras en el vocabulario: ')
#print(vocal_size)

# ENTRENAMIENTO DE LOS DATOS QUE PROPORCIONE

# Ajuste de los datos de entrenamiento al modelo creado
history = model.fit(X_train, train_labels, epochs=30, batch_size=8, verbose=1)

# Cálculo de los porcentajes de Eficiencia y Pérdida
score = model.evaluate(X_train, train_labels, verbose=1)
#print('\nTest Loss: ', score[0])
#print('\nTest Accuracy: ', score[1])

# GRAFICAR LA EFICIENCIA Y PERDIDA EN CADA EPOCA
'''
plt.figure(figsize=(12.5))
plt.ylim(-0.1,1.1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['loss'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Acc', 'Loss'])
plt.show()
'''

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


# MÓDULOS DE DETECCIÓN DE GRAMÁTICAS DÉBILES
Saludos_In = ["Hola", "Holi", "Cómo estás", "Que tal", "Como te va"]    
Despedidas_In = ["Adios", "Bye", "Hasta puego", "Nos pemos", "Hasta pronto"]
Gracias_In = ["Gracias", "Te agradezco", "Te doy las gracias"]
InsD = [Saludos_In, Despedidas_In, Gracias_In]

Saludos_Out = ["Hola, ¿Cómo estás?", "Un gusto de saludarte", "Me da gusto verte de nuevo", "Que pedo"]
Despedidas_Out  = ["Nos vemos, fue un gutso", "Que te vaya muy bien", "Regresa pronto, adios"]
Gracias_Out = ["Por nada, es un placer", "Me da mucho gusto poder ayudar", "Denada, para eso estoy"]
OutsD = [Saludos_Out, Despedidas_Out, Gracias_Out]

def Weak_gramars(inp):
    index = 0
    weak_act = 0
    for categoria in InsD:
        for gramatica in categoria:
            if inp.lower().count(gramatica.lower()) > 0:
                weak_act = 1
                print('\nChatBot: '+random.choice(OutsD[index]) + '  [Gramatica Débil]\n')
        index += 1
    return weak_act

# MÓDULOS DE  DETECCIÓN DE GRAMATICAS FUERTES
Insultos_In = ["Perra", "Puta", "Estúpida", "Maldita lisiada"]
Fan_In = ["Vikingos", "Breaking Bad", "Juego de Tronos"]
InsF = [Insultos_In, Fan_In]

Insultos_Out =["Tu lo serás", "¿Con esa boquita comes?", "Me ofendes", "Me la pelas"]
Fan_Out = ["Me encantan tus gustos", "Obra de arte", "Soy fan de ello"]
OutsF = [Insultos_Out, Fan_Out]

def Strong_grammars(inp):
    index = 0
    strong_act = 0
    for categoria in InsF:
        for gramatica in categoria:
            if inp.lower().count(gramatica.lower()) > 0:
                strong_act = 1 
                print('\nChatBot: '+random.choice(OutsF[index]) + '  [Gramatica Fuerte]\n')
        index += 1
    return strong_act

# RECONOCIMIENTO DE ENTIDADES
Paises = {'Mexico': 'Mexico', 'Francia':'Paris', 'España':'Madrid', 'Inglaterra':'Londres'}
Resp_Paises = ['Me gustaria poder visitarlo', 'Es hermoso ese lugar', 'Lastima que los chatbots no podemos ir a esos lugares :c '] 

Paises_Unknown = ['Lo siento, no conozco la capital de ese pais', 'Soy chafa, no se la respuesta', 'No c bro, disculpa']

def Country(inp):
    pais_act = 0
    for pais in Paises.keys():
        if inp.lower().count(pais.lower()) > 0:
            pais_act = 1 
            print('\nChatBot: '+'La capital de '+ str(pais) + ' es: ' + str(Paises.get(pais)) + ', '+ random.choice(Resp_Paises)+'\n')
    if pais_act == 0:
        print('\nChatbot: ' + random.choice(Paises_Unknown)+'\n')

# MODULO DE RECONOCIMIENTO DE ENTIDAD DE NUMEROS
Resp_Raiz = ['Verdad que soy muy listo', 'Soy la verga', 'Soy muy bueno en matemáticas']
Raiz_Unknown = ['Lo siento, me diste algún numero no valido', 'Puedes intentarlo con otro numero', 'No c bro, disculpa']


def Raiz(inp):
    num_act = 0 
    # Si en la frase hay un numero, que lo detecte y lo guarde
    num = re.search(r'(\d+)', inp.lower())
    if num != None:
        num_act = 1 
        print('\nChatBot: '+'La raiz cuadrada de '+num.group() + ' es: '+ str(round(math.sqrt(float(num.group())),4)) + ' ' + random.choice(Resp_Raiz) + '  [Entidad]\n')
    if num_act == 0:
        print('\nChatBot: '+random.choice(Raiz_Unknown)+'\n')        



# FUNCIÓN DE FUNCIONAMIENTO PARA EL CHATBOT 

def chat():
    print("\nChatBot: Hola soy un chatbot, comienza a hablar conmigo\n")
    while True:
        inp = input("   Tú: ")

        # Instrucción de fin de conversación (Cerrar el proceso)
        if inp.lower() == "salir":
            print("\nChatBot: Adios, vuelve prontooo\n")
            break

        # La gramatica fuerte es dominante (Si aparece no es necesario)
        # Evaluar la intención
        Strong = Strong_grammars(inp)
        if Strong == 0:

            # De cada entrada al sistema (inp), clasifica segun el modelo creado
            # y asigna un tag (Categoria)
            # Se usa argmax para regresar aquel que tiene mayor peso
            results = model.predict(Instancer(inp))
            results_index = np.argmax(results)
            tag = labels[results_index]

            # Valor de la clase con mayor score
            maxscore = np.max(results)
            print('Score del intent: '+ str(maxscore))

            # Con base en el tag se le asigna la intención del usuario
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']

            # Respuesta de la gramática débil
            weak = Weak_gramars(inp)

            # Elegir una respuesta aleatoria de la Response Pool (Si supera el umbral)
            # Umbral de desición
            if maxscore > 0.5:

                # Si se detecta una intención que esté asociada a entidades se envía a 
                # su respectivo módulo
                if tag == "Capital":
                    Country(inp)
                elif tag == "Raiz_Cuadrada":
                    Raiz(inp)
                else:
                    print('\nChatBot: '+ str(random.choice(responses)) + '[' + str(tag) + ']\n')
            else:
                if weak == 0:
                    print('\nChatBot: Lo siento, pero no comprendí. ¿Podrías responder de otra forma?\n')

print("Categorias del ChatBot: ")
print('Categorias: '+str(str(labels)+'\n'))


# ACTIVANDO EL CHATBOT
chat()