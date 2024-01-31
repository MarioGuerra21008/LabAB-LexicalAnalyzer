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

#Algoritmo de Thompson para convertir una expresión postfix en un AFN.

def is_letter_or_digit(char):
    return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9'

def regex_to_afn(regex,index):
    postfix = shunting_yard(regex)
    stack = []
    accept_state = []
    state_count = 0
    previous_symbol = None
    print(index)
    afn = nx.DiGraph()
    afn.add_node(state_count)
    start_state = state_count

    epsilon_state = state_count + 1
    afn.add_node(epsilon_state)
    afn.add_edge(state_count, epsilon_state, label='ε')
    state_count += 1

    # Mantén un seguimiento de los estados en cada nivel de alternancia
    alt_states = [set([1])]
    boolean = False
    cont = 0
    for symbol in postfix:

        if is_letter_or_digit(symbol):
            state_count += 1
            cont +=1
            afn.add_node(state_count)
            afn.add_edge(state_count - 1, state_count, label=symbol)
            #print("cont",cont)
            stack.append(state_count)
        elif symbol == '*':
            state = stack.pop()
            char = regex[state - 2]

            if char == '(':
              char = regex[state - 1]
            elif char == '*':
              char = regex[state - 7]
            elif char == 'ε':
              char = regex[state-1]

            if previous_symbol != '|':
               afn.add_edge(state, state, label=previous_symbol)
            afn.add_edge(state_count, state_count + 1, label='ε')

            if previous_symbol == '|' and cont <= 2:
              afn.add_edge(epsilon_state, state_count + 1, label='ε')

            epsilon_state = state_count + 2

            state_count += 1
            afn.add_node(state_count)
            stack.append(state_count)

            accept_state += [state]
            accept_state += [state - 1]

        elif symbol == '|':
            state2 = stack.pop()
            state1 = stack.pop()
            char1 = regex[state1 - 3]
            if char1 == ')':
              char1 = regex[state1 - 1]
            elif char1 == '*':
              char1 = regex[state1 - 1]
              boolean = True

            if char1 == '(':
              char1 = 'ε'

            char2 = regex[state2 - 3]
            if char2 == '(':
              char2 = regex[state2 ]

            if char2 == char1:
              char2 = regex[state2-2]
            elif char2 == '*':
              char2 == 'ε'

            state_count += 1

            if cont >= 3:
              afn.add_edge(epsilon_state+(cont-2), state1, label=char1)
              afn.add_edge(epsilon_state+(cont-2), state2, label=char2)

            else:
              afn.add_edge(epsilon_state, state1, label=char1)
              afn.add_edge(epsilon_state, state2, label=char2)
            print("num: ",cont)

            if index + 1 < len(postfix):
              sig = postfix[index+1]
            else:
              sig = None

            afn.add_edge(state1, state_count + 1, label='ε')
            afn.add_edge(state2, state_count + 1, label='ε')

            if sig == '*':
                #Transiciones Reflexivas
                afn.add_edge(state_count + 1, state1 - 1, label='ε')
                #afn.add_edge(state2, state2, label=char2)


            state_count += 1
            # Verifica si la arista existe antes de intentar eliminarla
            if afn.has_edge(state1, state2):
                afn.remove_edge(state1, state2)
            #afn.remove_edge(state1,state2)
            #afn.add_node(state_count)
            stack.append(state_count)
        elif symbol == '.':
            state2 = stack.pop()
            state1 = stack.pop()
            stack.append(state2)
        index += 1
        previous_symbol = symbol
    final_state = state_count +1
    accept_state += [final_state]
    #print("Estados de aceptacion: ",accept_state)
    afn.add_node(final_state)
    afn.add_edge(state_count, final_state, label='ε')

    afn.graph['start'] = start_state
    afn.graph['accept'] = accept_state

    return afn, accept_state

def compute_epsilon_closure(afn, state):
    epsilon_closure = set()
    stack = [state]

    while stack:
        current_state = stack.pop()
        epsilon_closure.add(current_state)

        for successor, edge_data in afn.adj[current_state].items():
            label = edge_data.get('label', None)
            if label == 'ε' and successor not in epsilon_closure:
                stack.append(successor)
    return epsilon_closure

def move(afn, state, symbol):
    target_states = set()
    for successor, edge_data in afn.adj[state].items():
        label = edge_data.get('label', None)
        if label == symbol:
            target_states.add(successor)
    return target_states

def get_alphabet(afn):
    alphabet = set()
    for _, _, label in afn.edges(data='label'):
        if label != 'ε':
            alphabet.add(label)
    return alphabet

def epsilon_closure(afn, states):
    closure = set(states)
    stack = list(states)
    while stack:
        state = stack.pop()
        for successor, attributes in afn[state].items():
            label = attributes['label']
            if label == 'ε' and successor not in closure:
                closure.add(successor)
                stack.append(successor)
            elif label == '*':
                closure.add(successor)
                for epsilon_successor in epsilon_closure(afn, {successor}):
                    if epsilon_successor not in closure:
                        closure.add(epsilon_successor)
                        stack.append(epsilon_successor)
    return closure

def check_membership(afn, s):
    current_states = epsilon_closure(afn, {afn.graph['start']})
    for symbol in s:
        next_states = set()

        for state in current_states:
            for successor, attributes in afn[state].items():
                if attributes['label'] == symbol:
                    next_states |= epsilon_closure(afn, {successor})
                    print("Estado actual: ",state)
                    print("Posibles caminos: ",afn[state])
                    print("Lee simbolo: ",symbol)
            if symbol != '*':
                current_states = next_states
    return any (state in afn.graph['accept'] for state in current_states)

if __name__ == "__main__":

    regex = input("Enter a regular expression: ")
    w = input("Enter a string to check: ")

    afn, accept_state = regex_to_afn(regex,0)
    print("afn edges",afn.edges(data='label'))

    # Obtener el conjunto de símbolos
    simbolos = set(label for _, _, label in afn.edges(data='label'))

    # Obtener el conjunto de estados iniciales
    estados_iniciales = {nodo for nodo in afn.nodes() if len(list(afn.predecessors(nodo))) == 0}
    estados_aceptacion = {nodo for nodo in afn.nodes() if len(list(afn.successors(nodo))) == 0}

    # Visualization
    pos = nx.spring_layout(afn, seed=42)
    labels = {edge: afn[edge[0]][edge[1]]['label'] for edge in afn.edges()}
    nx.draw_networkx_nodes(afn, pos)
    nx.draw_networkx_edges(afn, pos)
    nx.draw_networkx_edge_labels(afn, pos, edge_labels=labels)
    nx.draw_networkx_labels(afn, pos)

    plt.title("AFN Visualization")
    plt.axis("off")
    plt.figure(figsize=(30, 30))  # Ajusta el tamaño de la figura (ancho x alto) según tus preferencias
    plt.show()

    # Nombre del archivo de salida
    nombre_archivo = "descripcion_afn.txt"

    # Crear y escribir en el archivo de texto
    with open(nombre_archivo, "w") as archivo:
        archivo.write("ESTADOS = " + str(afn.nodes) + "\n")
        archivo.write("SIMBOLOS = " + str(simbolos) + "\n")
        archivo.write("INICIO = " + str(estados_iniciales) + "\n")
        archivo.write("ACEPTACION =" + str(accept_state) + "\n")
        archivo.write("TRANSICIONES =" + str(afn.edges(data='label')))

    print(f"Se ha creado el archivo '{nombre_archivo}' con la descripción del AFN.")
    #SIMULACION DEL AFN
    result = check_membership(afn, w)

    if result:
        print(f"'{w}' pertenece al lenguaje L({regex})")
    else:
        print(f"'{w}' no pertenece al lenguaje L({regex})")

#Algoritmo de Construcción de Subconjuntos para convertir un AFN en un AFD.

#Algoritmo de Construcción Directa para convertir una regex en un AFD.

#Algoritmo de Hopcroft para minimizar un AFD por medio de construcción de subconjuntos.
        
#Algoritmo para minimizar un AFD hecho por construcción directa.
