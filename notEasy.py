import random
# Librerias para los modulos de reconocimientos de entidades con numeros
import re 
import math
import webbrowser as wb

# MÓDULOS DE DETECCIÓN DE GRAMÁTICAS DÉBILES
Saludos_In = ["Hola", "Holi", "Cómo estás", "Que tal", "Como te va"]    
Despedidas_In = ["Adios", "Bye", "Hasta puego", "Nos pemos", "Hasta pronto"]
Gracias_In = ["Gracias", "Te agradezco", "Te doy las gracias"]
InsD = [Saludos_In, Despedidas_In, Gracias_In]

Saludos_Out = ["Hola, ¿Cómo estás?", "Un gusto de saludarte", "Me da gusto verte de nuevo", "Que pedo"]
Despedidas_Out  = ["Nos vemos, fue un gusto", "Que te vaya muy bien", "Regresa pronto, adios"]
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
Insultos_In = ["tonto", "sonso", "inutil", "feo"]
Fan_In = ["Vikingos", "Breaking Bad", "Juego de Tronos"]
InsF = [Insultos_In, Fan_In]

Insultos_Out =["Tu lo serás", "¿Con esa boquita comes?", "Me ofendes"]
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


# MODULO DE RECONOCIMIENTO DE ENTIDAD DE NUMEROS
Resp_Raiz = ['Verdad que soy muy listo', 'Soy muy bueno en matemáticas']
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

def pdff(inp):
    
    if inp:    
        print('\nChatBot: ')
        #wb.open_new('/home/gustavo/Escritorio/pdfff/CV_Gustavo_Rodriguez_Calzadaa.pdf')
        wb.open_new('/home/gustavo/Escritorio/pdfff/CV_Gustavo_Rodriguez_Calzadaa.pdf')
    else:
        print("No puedo abrir el documento")