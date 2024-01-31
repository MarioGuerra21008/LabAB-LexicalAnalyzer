#Universidad del Valle de Guatemala
#Diseño de Lenguajes de Programación - Sección: 10
#Mario Antonio Guerra Morales - Carné: 21008
#David Jonathan Aragón Vásquez - Carné: 21053
#Analizador Léxico por medio de una expresión regular.

#Importación de módulos y librerías para mostrar gráficamente los autómatas finitos.
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

#Algoritmo Shunting Yard para pasar una expresión infix a postfix.

def insert_concatenation(expression): #Función insert_concatenation para poder agregar los operadores al arreglo result.
    result = [] #Lista result para agregar los operadores.
    operators = "+|*()?" #Operadores en la lista.
    for i in range(len(expression)): #Por cada elemento según el rango de la variable expression:
        char = expression[i]
        result.append(char) #Se agrega caracter por caracter al arreglo.

        if i + 1 < len(expression): #Si la expresión es menor que la cantidad de elementos en el arreglo, se coloca en la posición i + 1.
            lookahead = expression[i + 1]

            if char.isalnum() and lookahead not in operators and lookahead != '.': #Si el caracter es una letra o un dígito, no está en los operadores y no es unc concatenación:
                result.append('.') #Agrega una concatenación a la lista result.
            elif char == '*' and lookahead.isalnum(): #Si el caracter es una estrella de Kleene o un signo de agrupación, agrega el punto como indica la notación postfix.
                result.append('.')
            elif char == ')' and lookahead.isalnum():
                result.append('.')
            elif char.isalnum() and lookahead == '(':
                result.append('.')
            elif char == ')' and lookahead == '(':
                result.append('.')

    return ''.join(result) #Devuelve el resultado.

def shunting_yard(expression): #Función para realizar el algoritmo shunting yard.
     precedence = {'+': 1, '|': 1, '.': 2, '*': 3, '?':3} # Orden de precedencia entre operadores.

     output_queue = [] #Lista de salida como notación postfix.
     operator_stack = []
     i = 0 #Inicializa contador.

     expression = insert_concatenation(expression) #Llama a la función para que se ejecute.

     while i < len(expression): #Mientras i sea menor que la longitud de la expresión.
         token = expression[i] #El token es igual al elemento en la lista en la posición i.
         if token.isalnum() or token == 'ε': #Si el token es una letra o un dígito, o es epsilon.
             output_queue.append(token) #Se agrega a output_queue.
         elif token in "+|*.?": #Si hay alguno de estos operadores en el token:
             while (operator_stack and operator_stack[-1] != '(' and #Se toma en cuenta la precedencia y el orden de operadores para luego añadirla al queue y a la pila.
                    precedence[token] <= precedence.get(operator_stack[-1], 0)):
                 output_queue.append(operator_stack.pop())
             operator_stack.append(token)
         elif token == '(': #Si el token es una apertura de paréntesis se añade a la pila de operadores.
             operator_stack.append(token)
         elif token == ')': #Si el token es una cerradura de paréntesis se añade al queue y pila de operadores, se ejecuta un pop en ambas.
             while operator_stack and operator_stack[-1] != '(':
                 output_queue.append(operator_stack.pop())
             operator_stack.pop()
         elif token == '.': #Si el token es un punto o concatenación se realiza un pop en la pila y se añade al output_queue.
             while operator_stack and operator_stack[-1] != '(':
                 output_queue.append(operator_stack.pop())
             if operator_stack[-1] == '(':
                 operator_stack.pop()
         i += 1 #Suma uno al contador.

     while operator_stack: #Mientras se mantenga el operator_stack, por medio de un pop se agregan los elementos al output_queue.
         output_queue.append(operator_stack.pop())

     if not output_queue: #Si no hay un queue de salida, devuelve epsilon.
         return 'ε'
     else: #Si hay uno, lo muestra en pantalla.
         return ''.join(output_queue)

expression = input("Enter your infix expression: ") #Entrada para colocar la expresión en infix.
postfix_expression = shunting_yard(expression) #Se declara postfix_expression para el método shunting yard para ejecutar el algoritmo.
print("Postfix expression:", postfix_expression) #Mensaje de salida con la expresión transformada a postfix.

