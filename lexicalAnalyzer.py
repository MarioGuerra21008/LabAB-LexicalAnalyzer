#Universidad del Valle de Guatemala
#Diseño de Lenguajes de Programación - Sección: 10
#Mario Antonio Guerra Morales - Carné: 21008
#David Jonathan Aragón Vásquez - Carné: 21053
#Analizador Léxico por medio de una expresión regular.

#Importación de módulos y librerías para mostrar gráficamente los autómatas finitos.
import networkx as nx
import matplotlib.pyplot as plt
from collections import deque, defaultdict

#Algoritmo Shunting Yard para pasar una expresión infix a postfix.

def insert_concatenation(expression): #Función insert_concatenation para poder agregar los operadores al arreglo result.
    result = [] #Lista result para agregar los operadores.
    operators = "+|*()?" #Operadores en la lista.
    for i in range(len(expression)): #Por cada elemento según el rango de la variable expression:
        char = expression[i]
        result.append(char) #Se agrega caracter por caracter al arreglo.

        if i + 1 < len(expression): #Si la expresión es menor que la cantidad de elementos en el arreglo, se coloca en la posición i + 1.
            lookahead = expression[i + 1]

            if char.isalnum() or char == 'ε':
                if lookahead not in operators and lookahead != '.': #Si el caracter es una letra o un dígito, no está en los operadores y no es unc concatenación:
                    result.append('.') #Agrega una concatenación a la lista result.
            elif char == '*' and lookahead.isalnum(): #Si el caracter es una estrella de Kleene o un signo de agrupación, agrega el punto como indica la notación postfix.
                result.append('.')
            elif char == ')' and lookahead.isalnum():
                result.append('.')
            elif char.isalnum() and lookahead == '(':
                result.append('.')
            elif char == ')' and lookahead == '(':
                result.append('.')
            elif char == '?' and lookahead.isalnum():
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
         elif token in "+|*?.": #Si hay alguno de estos operadores en el token:
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

#Algoritmo de Thompson para convertir una expresión postfix en un AFN.

def is_letter_or_digit(char): #Función que sirve para detectar si es una letra o un dígito en la expresión regular.
    return 'a' <= char <= 'z' or 'A' <= char <= 'Z' or '0' <= char <= '9'

def regex_to_afn(regex,index):
    #Convertir la expresión regular a notación postfix
    postfix = shunting_yard(regex)

    #Inicialización de variables
    stack = []  #Pila para mantener un seguimiento de los estados
    accept_state = []  #Lista de estados de aceptación
    state_count = 0  #Contador de estados
    previous_symbol = None  #Símbolo anterior en el proceso

    #Crear un grafo dirigido para representar el AFN
    afn = nx.DiGraph()
    afn.add_node(state_count)
    start_state = state_count  #Estado inicial del AFN
    epsilon_state = state_count + 1  #Estado de transición épsilon
    afn.add_node(epsilon_state)
    afn.add_edge(state_count, epsilon_state, label='ε')  #Transición épsilon desde el estado inicial
    state_count += 1
    alt_states = [set([1])]  #Conjuntos de estados alternativos
    cont = 0  #Contador auxiliar
    
    #Recorrer la expresión regular en notación postfix
    for symbol in postfix:
        #Manejar los símbolos y operadores de la expresión regular
        if is_letter_or_digit(symbol):  #Si el símbolo es una letra o un dígito
            state_count += 1
            cont += 1
            afn.add_node(state_count)
            afn.add_edge(state_count - 1, state_count, label=symbol)
            stack.append(state_count)
        elif symbol == '*':  #Operador de clausura de Kleene
            #Manejar la clausura de Kleene
            state = stack.pop()
            char = regex[state - 2]  #Obtener el carácter anterior al estado actual

            #Ajustar el carácter dependiendo del contexto
            if char == '(':
                char = regex[state - 1]
            elif char == '*':
                char = regex[state - 7]
            elif char == 'ε':
                char = regex[state - 1]

            #Agregar transición desde el estado actual al próximo estado
            if previous_symbol != '|':
                afn.add_edge(state, state, label=previous_symbol)
            afn.add_edge(state_count, state_count + 1, label='ε')

            #Manejar la clausura de Kleene y actualizar el estado actual
            if previous_symbol == '|' and cont <= 2:
                afn.add_edge(epsilon_state, state_count + 1, label='ε')

            epsilon_state = state_count + 2
            state_count += 1
            afn.add_node(state_count)
            stack.append(state_count)

            #Agregar estados de aceptación
            accept_state += [state]
            accept_state += [state - 1]

        elif symbol == '|':  #Operador de alternancia
            #Manejar la alternancia de estados
            state2 = stack.pop()
            state1 = stack.pop()
            char1 = regex[state1 - 3]  #Obtener el carácter anterior al primer estado

            #Ajustar el carácter dependiendo del contexto
            if char1 == ')':
                char1 = regex[state1 - 1]
            elif char1 == '*':
                char1 = regex[state1 - 1]

            if char1 == '(':
                char1 = 'ε'

            char2 = regex[state2 - 3]  #Obtener el carácter anterior al segundo estado
            if char2 == '(':
                char2 = regex[state2]

            if char2 == char1:
                char2 = regex[state2 - 2]
            elif char2 == '*':
                char2 == 'ε'

            state_count += 1

            #Agregar transiciones desde el estado épsilon a los estados alternativos
            if cont >= 3:
                afn.add_edge(epsilon_state + (cont - 2), state1, label=char1)
                afn.add_edge(epsilon_state + (cont - 2), state2, label=char2)
            else:
                afn.add_edge(epsilon_state, state1, label=char1)
                afn.add_edge(epsilon_state, state2, label=char2)

            #Actualizar el siguiente símbolo en el proceso
            if index + 1 < len(postfix):
                sig = postfix[index + 1]
            else:
                sig = None

            #Agregar transiciones desde los estados a un nuevo estado
            afn.add_edge(state1, state_count + 1, label='ε')
            afn.add_edge(state2, state_count + 1, label='ε')

            if sig == '*':
                afn.add_edge(state_count + 1, state1 - 1, label='ε')
            state_count += 1
            if afn.has_edge(state1, state2):
                afn.remove_edge(state1, state2)
            stack.append(state_count)
        elif symbol == '.':  #Operador de concatenación
            state2 = stack.pop()
            state1 = stack.pop()
            stack.append(state2)
        elif symbol == '?':  # Operador de cero o una ocurrencia
            # Manejar el operador '?'
            state1 = stack.pop()
            state2 = state_count + 1
            afn.add_node(state2)
            afn.add_edge(state1, state2, label='ε')  # Transición desde el estado antes del '?'
            afn.add_edge(state1, state2 + 1, label='ε')  # Transición ε directa al siguiente estado
            afn.add_edge(state2, state2 + 1, label=previous_symbol)  # Transición desde el estado del '?'
            afn.add_edge(state1, state2 + 1, label='ε')
            stack.append(state2 + 1)
            state_count += 1
        index += 1
        previous_symbol = symbol  #Actualizar el símbolo anterior en el proceso
    
    #Establecer el estado final y los estados de aceptación
    final_state = state_count + 1
    accept_state += [final_state]
    afn.add_node(final_state)
    afn.add_edge(state_count, final_state, label='ε')

    #Establecer el estado inicial y los estados de aceptación en el grafo
    afn.graph['start'] = start_state
    afn.graph['accept'] = accept_state

    #Retornar el AFN generado junto con los estados de aceptación
    return afn, accept_state

def compute_epsilon_closure(afn, state):
    #Inicializar conjunto de cierre épsilon y pila con el estado inicial
    epsilon_closure = set()
    stack = [state]

    #Recorrer el grafo del AFN
    while stack:
        #Sacar un estado de la pila
        current_state = stack.pop()
        #Agregarlo al cierre épsilon
        epsilon_closure.add(current_state)

        #Recorrer los sucesores del estado actual
        for successor, edge_data in afn.adj[current_state].items():
            label = edge_data.get('label', None)
            #Si la etiqueta es épsilon y el sucesor no está en el cierre épsilon, agregarlo a la pila
            if label == 'ε' and successor not in epsilon_closure:
                stack.append(successor)
    #Retornar el cierre épsilon
    return epsilon_closure

def move(afn, state, symbol):
    #Inicializar conjunto de estados destino
    target_states = set()
    
    #Recorrer los sucesores del estado actual
    for successor, edge_data in afn.adj[state].items():
        label = edge_data.get('label', None)
        #Si la etiqueta coincide con el símbolo, agregar el sucesor al conjunto de estados destino
        if label == symbol:
            target_states.add(successor)
    #Retornar los estados destino
    return target_states

def get_alphabet(afn):
    #Inicializar conjunto de símbolos del alfabeto
    alphabet = set()
    
    #Recorrer todas las aristas del grafo del AFN
    for _, _, label in afn.edges(data='label'):
        #Si la etiqueta no es épsilon, agregarla al alfabeto
        if label != 'ε':
            alphabet.add(label)
    #Retornar el alfabeto
    return alphabet

def epsilon_closure(afn, states):
    #Inicializar cierre épsilon con los estados dados y una pila
    closure = set(states)
    stack = list(states)
    
    #Recorrer la pila
    while stack:
        state = stack.pop()
        #Recorrer los sucesores del estado actual
        for successor, attributes in afn[state].items():
            label = attributes['label']
            #Si la etiqueta es épsilon y el sucesor no está en el cierre épsilon, agregarlo al cierre y a la pila
            if label == 'ε' and successor not in closure:
                closure.add(successor)
                stack.append(successor)
            #Si la etiqueta es '*', agregar el sucesor al cierre y expandir su cierre épsilon
            elif label == '*':
                closure.add(successor)
                for epsilon_successor in epsilon_closure(afn, {successor}):
                    if epsilon_successor not in closure:
                        closure.add(epsilon_successor)
                        stack.append(epsilon_successor)
    #Retornar el cierre épsilon
    return closure

def check_membership(afn, s):
    #Inicializar estados actuales con el cierre épsilon del estado inicial
    current_states = epsilon_closure(afn, {afn.graph['start']})
    
    #Recorrer los símbolos de la cadena de entrada
    for symbol in s:
        next_states = set()
        #Recorrer los estados actuales
        for state in current_states:
            #Recorrer los sucesores del estado actual
            for successor, attributes in afn[state].items():
                if attributes['label'] == symbol:
                    #Si la etiqueta coincide con el símbolo, agregar el cierre épsilon del sucesor a los estados siguientes
                    next_states |= epsilon_closure(afn, {successor})
                    print("Estado actual: ",state)
                    print("Posibles caminos: ",afn[state])
                    print("Lee simbolo: ",symbol)
            #Actualizar los estados actuales con los siguientes estados
            if symbol != '*':
                current_states = next_states
    #Verificar si algún estado actual es un estado de aceptación
    return any(state in afn.graph['accept'] for state in current_states)

#Algoritmo de Construcción de Subconjuntos para convertir un AFN en un AFD.

def afn_to_afd(afn):
    # Obtener el estado inicial del AFN
    start_state = afn.graph['start']
    # Calcular el cierre épsilon del estado inicial
    epsilon_closure = compute_epsilon_closure(afn, start_state)

    # Inicializar un grafo dirigido para representar el AFD
    dfa = nx.DiGraph()
    # Convertir el cierre épsilon del estado inicial en una tupla ordenada
    dfa_start_state = tuple(sorted(epsilon_closure))
    # Agregar el estado inicial al AFD
    dfa.add_node(dfa_start_state)

    # Inicializar una lista de estados no marcados con el estado inicial del AFD
    unmarked_states = [dfa_start_state]

    # Proceso de construcción del AFD
    while unmarked_states:
        # Tomar un estado no marcado del AFD
        current_dfa_state = unmarked_states.pop()

        # Para cada símbolo del alfabeto del AFN
        for symbol in get_alphabet(afn):
            # Calcular los estados a los que se llega desde el estado actual del AFD utilizando el símbolo
            target_states = set()
            for afn_state in current_dfa_state:
                target_states.update(move(afn, afn_state, symbol))

            # Calcular el cierre épsilon de los estados obtenidos
            epsilon_closure_target = set()
            for target_state in target_states:
                epsilon_closure_target.update(compute_epsilon_closure(afn, target_state))

            # Convertir el cierre épsilon de los estados en una tupla ordenada
            dfa_target_state = tuple(sorted(epsilon_closure_target))

            # Si el estado obtenido no está en el AFD, marcarlo como no marcado y agregarlo al AFD
            if dfa_target_state not in dfa:
                unmarked_states.append(dfa_target_state)
                dfa.add_node(dfa_target_state)
            # Agregar una transición desde el estado actual del AFD al estado obtenido con el símbolo actual
            dfa.add_edge(current_dfa_state, dfa_target_state, label=symbol)

    # Establecer el estado inicial del AFD
    dfa.graph['start'] = dfa_start_state
    # Obtener los estados de aceptación del AFD
    dfa_accept_states = [state for state in dfa.nodes if any(afn_state in afn.graph['accept'] for afn_state in state)]
    # Establecer los estados de aceptación del AFD
    dfa.graph['accept'] = dfa_accept_states
    # Retornar el AFD construido
    return dfa

#Algoritmo de Construcción Directa para convertir una regex en un AFD.

class Nodo:
    def __init__(self, valor):
        self.valor = valor
        self.nullable = False
        self.firstpos = set()
        self.lastpos = set()

def nullable(treeDC):
    if isinstance(treeDC, list):
        if treeDC[0] == '.':
            return nullable(treeDC[1]) and nullable(treeDC[2])
        elif treeDC[0] == '*':
            return True
    return False

def firstpos(treeDC):
    if isinstance(treeDC, list):
        if treeDC[0] == '.':
            if nullable(treeDC[1]):
                return firstpos(treeDC[1]).union(firstpos(treeDC[2]))
            else:
                return firstpos(treeDC[1])
        elif treeDC[0] == '*':
            return firstpos(treeDC[1])
        pass

def lastpos(treeDC):
    if isinstance(treeDC, list):
        if treeDC[0] == '.':
            if nullable(treeDC[2]):
                return lastpos(treeDC[1]).union(lastpos(treeDC[2]))
            else:
                return lastpos(treeDC[2])
        elif treeDC[0] == '*':
            return lastpos(treeDC[1])
        pass


def followpos(treeDC, followpos_dict):
    print(treeDC)
    if isinstance(treeDC, list) and treeDC[0] == '.':
        for pos in lastpos(treeDC[1]):
            followpos_dict[pos].update(firstpos(treeDC[2]))
        followpos(treeDC[2], followpos_dict)
    elif isinstance(treeDC, list) and treeDC[0] == '*':
        for pos in lastpos(treeDC[1]):
            followpos_dict[pos].update(firstpos(treeDC[1]))
        followpos(treeDC[1], followpos_dict)
    pass

def build_nfa(regex):
    stack = []
    pos = 1
    followpos_dict = defaultdict(set)

    for char in regex:
        if char == '(':
            stack.append(char)
        elif char == ')':
            stack.pop()
        elif char == '|':
            stack.append(char)
        elif char == '*':
            treeDC = ('*', stack.pop())
            nullable_val = nullable(treeDC)
            firstpos_val = firstpos(treeDC)
            lastpos_val = lastpos(treeDC)
            for pos in lastpos_val:
                followpos_dict[pos].update(firstpos_val)
            stack.append(pos)
            pos += 1
        else:
            stack.append(pos)
            pos += 1

    root = stack[0]

    followpos(root, followpos_dict)

    return root, followpos_dict

def epsilon_closureDC(estado, followpos_dict):
    epsilon_cerradura = set([estado])
    stack = [estado]
    while stack:
        current = stack.pop()
        for siguiente in followpos_dict[current]:
            if siguiente not in epsilon_cerradura:
                epsilon_cerradura.add(siguiente)
                stack.append(siguiente)
    return epsilon_cerradura

def moveDC(epsilon_cerradura, simbolo, followpos_dict):
    move_result = set()
    for estado in epsilon_cerradura:
        move_result.update(followpos_dict[estado] if estado[0] == simbolo else set())
    return move_result

def build_dfa(nfa_root, followpos_dict):
    alfabeto = set()
    for conjunto in followpos_dict.values():
        alfabeto.update(conjunto)

    dfa_transiciones = {}
    estados_no_marcados = [epsilon_closureDC(nfa_root, followpos_dict)]
    estados_marcados = []

    while estados_no_marcados:
        current = estados_no_marcados.pop()
        estados_marcados.append(current)

        for simbolo in alfabeto:
            siguiente_estado = epsilon_closureDC(move(current, simbolo, followpos_dict), followpos_dict)
            if siguiente_estado not in estados_marcados + estados_no_marcados:
                estados_no_marcados.append(siguiente_estado)
            dfa_transiciones[(tuple(current), simbolo)] = tuple(siguiente_estado)

    return dfa_transiciones

def visualize_dfa(dfa_transiciones):
    G = nx.DiGraph()
    for estado, transicion in dfa_transiciones.items():
        G.add_edge(estado[0], transicion, label=estado[1])
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, arrows=True)
    edge_labels = {(estado, transicion): label for (estado, transicion), label in nx.get_edge_attributes(G, 'label').items()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.show()

def mainConstruccionDirecta():
    regex = input("Ingrese la expresión regular: ")
    nfa_root, followpos_dict = build_nfa(regex)
    dfa_transiciones = build_dfa(nfa_root, followpos_dict)
    visualize_dfa(dfa_transiciones)
    


#Algoritmo de Hopcroft para minimizar un AFD por medio de construcción de subconjuntos.

def hopcroft_minimization(dfa):
    # Inicializar particiones con estados de aceptación y no de aceptación
    partitions = [dfa.graph['accept'], list(set(dfa.nodes) - set(dfa.graph['accept']))]
    # Inicializar una lista de trabajo con la partición de estados de aceptación
    worklist = deque([dfa.graph['accept']])

    # Proceso de minimización de Hopcroft
    while worklist:
        partition = worklist.popleft()
        for symbol in get_alphabet(dfa):
            divided_partitions = []
            for p in partitions:
                divided = set()
                for state in p:
                    # Verificar si hay transiciones con el símbolo actual hacia estados en la partición
                    successors = set(dfa.successors(state))
                    if symbol in [dfa.edges[(state, succ)]['label'] for succ in successors]:
                        divided.add(state)
                if divided:
                    divided_partitions.append(divided)
                    if len(divided) < len(p):
                        divided_partitions.append(list(set(p) - divided))
            # Actualizar las particiones si se dividen en particiones más pequeñas
            if len(divided_partitions) > len(partitions):
                if partition in partitions:
                    partitions.remove(partition)
                partitions.extend(divided_partitions)
                worklist.extend(divided_partitions)
    # Crear el DFA minimizado
    min_dfa = nx.DiGraph()
    state_mapping = {}

    # Mapear estados a su representación en la partición
    for i, partition in enumerate(partitions):
        if partition:
            min_state = ', '.join(sorted(str(state) for state in partition))
            state_mapping.update({state: min_state for state in partition})

    # Construir las transiciones del DFA minimizado
    for source, target, label in dfa.edges(data='label'):
        min_source = state_mapping[source]
        min_target = state_mapping[target]
        min_dfa.add_edge(min_source, min_target, label=label)

    # Establecer el estado inicial y los estados de aceptación del DFA minimizado
    min_dfa.graph['start'] = state_mapping[dfa.graph['start']]
    min_dfa.graph['accept'] = [state_mapping[state] for state in dfa.graph['accept'] if state in state_mapping]

    # Remover nodos y aristas no alcanzables del DFA minimizado
    if '()' in min_dfa.nodes:
        min_dfa.remove_node('()')
        for source, target in list(min_dfa.edges):
            if target == '()':
                min_dfa.remove_edge(source, target)
    # Retornar el DFA minimizado
    return min_dfa

def remove_unreachable_states(dfa):
    # Encontrar estados alcanzables desde el estado inicial
    reachable_states = set()
    stack = [dfa.graph['start']]

    while stack:
        state = stack.pop()
        if state not in reachable_states:
            reachable_states.add(state)
            stack.extend(successor for successor in dfa.successors(state))
    # Encontrar estados no alcanzables
    unreachable_states = set(dfa.nodes) - reachable_states
    # Remover estados no alcanzables
    dfa.remove_nodes_from(unreachable_states)

#Algoritmo para minimizar un AFD hecho por construcción directa.



#Simulación:

if __name__ == "__main__":
    expression = input("Enter your infix expression: ") 
    postfix_expression = shunting_yard(expression) 
    print("Postfix expression:", postfix_expression) 

    regex = input("Enter a regular expression: ")
    w = input("Enter a string to check: ")

    afn, accept_state = regex_to_afn(regex, 0)
    print("afn edges", afn.edges(data='label'))

    # Obtener el conjunto de símbolos
    simbolos = set(label for _, _, label in afn.edges(data='label'))

    # Obtener el conjunto de estados iniciales
    estados_iniciales = {nodo for nodo in afn.nodes() if len(list(afn.predecessors(nodo))) == 0}
    estados_aceptacion = {nodo for nodo in afn.nodes() if len(list(afn.successors(nodo))) == 0}

    # Visualización AFN
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    pos = nx.spring_layout(afn, seed=42)
    labels = {edge: afn[edge[0]][edge[1]]['label'] for edge in afn.edges()}
    nx.draw_networkx_nodes(afn, pos, node_color='blue')
    nx.draw_networkx_edges(afn, pos)
    nx.draw_networkx_edge_labels(afn, pos, edge_labels=labels)
    nx.draw_networkx_labels(afn, pos)
    plt.title("AFN Visualization")
    plt.axis("off")

    plt.show()

    # SIMULACION DEL AFN
    result = check_membership(afn, w)
    if result:
        print(f"'{w}' pertenece al lenguaje L({regex})")
    else:
        print(f"'{w}' no pertenece al lenguaje L({regex})")

    # Convierte el AFN a AFD
    afd = afn_to_afd(afn)
    # Elimina el estado final vacío '()' y sus aristas del AFD
    if ((), ()) in afd.nodes:
        afd.remove_node(((), ()))
        # Asegúrate de también eliminar cualquier arista que apunte a este nodo
        for source, target in list(afd.edges):
            if target == ((), ()):
                afd.remove_edge(source, target)

    filtered_edges = [(source, target, label) for source, target, label in afd.edges(data='label') if source != () and target != ()]

    # Filtrar los nodos que no son tuplas vacías
    filtered_nodes = [node for node in afd.nodes if node != ()]

    simbolos = set(label for _, _, label in afd.edges(data='label'))
    # Obtener el conjunto de estados iniciales
    estados_iniciales = {nodo for nodo in filtered_nodes if len(list(afd.predecessors(nodo))) == 0}

    estados_aceptacion = set()
    for nodo in filtered_nodes:
        aceptacion = True
        for succ in afd.successors(nodo):
            if succ not in filtered_nodes or succ == ():
                aceptacion = False
                break
        if aceptacion:
            estados_aceptacion.add(nodo)

    G = nx.DiGraph()

    # Agregar nodos a filtered_graph
    for source, target, label in filtered_edges:
        G.add_node(source)
        G.add_node(target)
        G.add_edge(source, target, label=label)

    # Obtener las posiciones de los nodos para el dibujo
    pos = nx.spring_layout(G)

    # Dibujar los nodos y las aristas
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    labels = {edge: label for edge, label in nx.get_edge_attributes(G, 'label').items()}
    nx.draw(G, pos, with_labels=True, node_size=200, node_color='blue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("AFD Visualization")
    plt.axis("off")

    plt.show()

    # SIMULACION DEL AFD
    result = check_membership(afd, w)
    if result:
        print(f"'{w}' pertenece al lenguaje L({regex})")
    else:
        print(f"'{w}' no pertenece al lenguaje L({regex})")

    #Minimiza el AFD
    remove_unreachable_states(afd)

    min_dfa = hopcroft_minimization(afd)

    # Elimina el estado final vacío '()' y sus aristas del AFD minimizado
    if '()' in min_dfa.nodes:
        min_dfa.remove_node('()')

        # Asegúrate de también eliminar cualquier arista que apunte a este nodo
        for source, target in list(min_dfa.edges):
            if target == '()':
                min_dfa.remove_edge(source, target)

    # Visualización AFD minimizado
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    pos_min = nx.spring_layout(min_dfa)
    nx.draw(min_dfa, pos_min, with_labels=True, node_size=200, node_color='blue')
    plt.title("Minimized DFA Visualization")
    plt.axis("off")

    plt.show()

    # SIMULACION DEL AFD MINIMIZADO
    result_min = check_membership(min_dfa, w)
    if result_min:
        print(f"'{w}' pertenece al lenguaje L({regex})")
    else:
        print(f"'{w}' no pertenece al lenguaje L({regex})")

    #Construcción directa (AFD).
    
    mainConstruccionDirecta()
