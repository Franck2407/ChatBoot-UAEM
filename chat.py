from Train import *
from notEasy import *

# FUNCIÓN PARA EL FUNCIONAMIENTO CHATBOT 
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
            for tg in data["intentos"]:
                if tg['clave'] == tag:
                    responses = tg['respuestas']

            # Respuesta de la gramática débil
            weak = Weak_gramars(inp)

            # Elegir una respuesta aleatoria de la Response Pool (Si supera el umbral)
            # Umbral de desición
            if maxscore > 0.5:

                # Si se detecta una intención que esté asociada a entidades se envía a 
                # su respectivo módulo
                if tag == "Raiz_Cuadrada":
                    Raiz(inp)
                elif tag == "Pdfff":
                    pdff(inp)
                else:
                    print('\nChatBot: '+ str(random.choice(responses)) + '[' + str(tag) + ']\n')

print("Categorias del ChatBot: ")
print('Categorias: '+str(str(labels)+'\n'))

# ACTIVANDO EL CHATBOT
chat()