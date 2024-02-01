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

def afn_to_afd(afn):
    start_state = afn.graph['start']
    epsilon_closure = compute_epsilon_closure(afn, start_state)

    dfa = nx.DiGraph()
    dfa_start_state = tuple(sorted(epsilon_closure))
    dfa.add_node(dfa_start_state)

    unmarked_states = [dfa_start_state]

    while unmarked_states:
        current_dfa_state = unmarked_states.pop()

        for symbol in get_alphabet(afn):
            target_states = set()
            for afn_state in current_dfa_state:
                target_states.update(move(afn, afn_state, symbol))

            epsilon_closure_target = set()
            for target_state in target_states:
                epsilon_closure_target.update(compute_epsilon_closure(afn, target_state))

            dfa_target_state = tuple(sorted(epsilon_closure_target))

            if dfa_target_state not in dfa:
                unmarked_states.append(dfa_target_state)
                dfa.add_node(dfa_target_state)

            dfa.add_edge(current_dfa_state, dfa_target_state, label=symbol)

    dfa.graph['start'] = dfa_start_state

    # Determine accept states in DFA
    dfa_accept_states = [state for state in dfa.nodes if any(afn_state in afn.graph['accept'] for afn_state in state)]
    dfa.graph['accept'] = dfa_accept_states
    #print("nodes:",dfa.edges)
    return dfa

if __name__ == "__main__":
  #AFD
    w = input("Enter a string to check: ")

    #Convierte el afd a afn
    afd = afn_to_afd(afn)
    # Elimina el estado final vacío '()' y sus aristas del AFD
    if ((), ()) in afd.nodes:
        afd.remove_node(((), ()))

        # Asegúrate de también eliminar cualquier arista que apunte a este nodo
        for source, target in list(afd.edges):
            if target == ((), ()):
                afd.remove_edge(source, target)

    # Filtrar las aristas que no tienen tuplas vacías en ambos extremos


    filtered_edges = [(source, target, label) for source, target, label in afd.edges(data='label') if source != () and target != ()]

    # Filtrar los nodos que no son tuplas vacías
    filtered_nodes = [node for node in afd.nodes if node != ()]


    #print("nodes:",afd.nodes)

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
        #print("grafo G:", G )

    # Obtener las posiciones de los nodos para el dibujo
    pos = nx.spring_layout(G)

    # Dibujar los nodos y las aristas
    labels = {edge: label for edge, label in nx.get_edge_attributes(G, 'label').items()}
    nx.draw(G, pos, with_labels=True, node_size=800, node_color='lightblue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("AFD Visualization")
    plt.figure(figsize=(18, 16))  # Ajusta el tamaño de la figura (ancho x alto) según tus preferencias
    plt.show()

    # Nombre del archivo de salida
    nombre_archivo = "descripcion_afd.txt"


    # Crear y escribir en el archivo de texto
    with open(nombre_archivo, "w") as archivo:
        archivo.write("ESTADOS = " + str(filtered_nodes) + "\n")
        archivo.write("SIMBOLOS = " + str(simbolos) + "\n")
        archivo.write("INICIO = " + str(estados_iniciales) + "\n")
        archivo.write("ACEPTACION =" + str(estados_aceptacion) + "\n")
        archivo.write("TRANSICIONES =" + str(filtered_edges))

    #SIMULACION DEL AFD
    result = check_membership(afd, w)

    if result:
        print(f"'{w}' belongs to L({regex})")
    else:
        print(f"'{w}' does not belong to L({regex})")

#Algoritmo de Construcción Directa para convertir una regex en un AFD.



#Algoritmo de Hopcroft para minimizar un AFD por medio de construcción de subconjuntos.

def hopcroft_minimization(dfa):
    # Inicialización
    partitions = [dfa.graph['accept'], list(set(dfa.nodes) - set(dfa.graph['accept']))]
    worklist = deque([dfa.graph['accept']])

    while worklist:
        partition = worklist.popleft()

        for symbol in get_alphabet(dfa):
            # Dividir la partición actual en subparticiones
            divided_partitions = []
            for p in partitions:
                divided = set()
                for state in p:
                    successors = set(dfa.successors(state))
                    if symbol in [dfa.edges[(state, succ)]['label'] for succ in successors]:
                        divided.add(state)
                if divided:
                    divided_partitions.append(divided)
                    if len(divided) < len(p):
                        divided_partitions.append(list(set(p) - divided))

            if len(divided_partitions) > len(partitions):
                # Comprueba si la partición actual todavía está en la lista antes de eliminarla
                if partition in partitions:
                  partitions.remove(partition)
                partitions.extend(divided_partitions)
                worklist.extend(divided_partitions)

    # Construir el AFD minimizado
    min_dfa = nx.DiGraph()
    state_mapping = {}  # Mapeo de estados originales a estados minimizados

    for i, partition in enumerate(partitions):
        if partition:
            min_state = ', '.join(sorted(str(state) for state in partition))  # Nuevo nombre de estado como contenido de los estados
            state_mapping.update({state: min_state for state in partition})

    for source, target, label in dfa.edges(data='label'):
        min_source = state_mapping[source]
        min_target = state_mapping[target]
        min_dfa.add_edge(min_source, min_target, label=label)

    min_dfa.graph['start'] = state_mapping[dfa.graph['start']]
    min_dfa.graph['accept'] = [state_mapping[state] for state in dfa.graph['accept'] if state in state_mapping]

    # Elimina el estado final vacío '()' y sus aristas del AFD
    if '()' in min_dfa.nodes:
        min_dfa.remove_node('()')

        # Asegúrate de también eliminar cualquier arista que apunte a este nodo
        for source, target in list(min_dfa.edges):
            if target == '()':
                min_dfa.remove_edge(source, target)

    return min_dfa

def remove_unreachable_states(dfa):
    # Realiza un análisis de alcanzabilidad desde el estado inicial del DFA
    reachable_states = set()
    stack = [dfa.graph['start']]

    while stack:
        state = stack.pop()
        if state not in reachable_states:
            reachable_states.add(state)
            stack.extend(successor for successor in dfa.successors(state))

    # Elimina los estados inalcanzables y las transiciones asociadas
    unreachable_states = set(dfa.nodes) - reachable_states
    dfa.remove_nodes_from(unreachable_states)

if __name__ == "__main__":
    # AFD
    w = input("Enter a string to check: ")

    remove_unreachable_states(afd)

    # Minimiza el AFD
    min_dfa = hopcroft_minimization(afd)

    # Elimina el estado final vacío '()' y sus aristas del AFD minimizado
    if '()' in min_dfa.nodes:
        min_dfa.remove_node('()')

        # Asegúrate de también eliminar cualquier arista que apunte a este nodo
        for source, target in list(min_dfa.edges):
            if target == '()':
                min_dfa.remove_edge(source, target)

    # Dibujar el AFD minimizado
    G_min = nx.DiGraph()

    # Agregar nodos y aristas al grafo minimizado
    for source, target, label in min_dfa.edges(data='label'):
        G_min.add_node(source)
        G_min.add_node(target)
        G_min.add_edge(source, target, label=label)

    # Obtener las posiciones de los nodos para el dibujo
    pos_min = nx.spring_layout(G_min)

    # Dibujar los nodos y las aristas del AFD minimizado, evitando las transiciones vacías
    labels_min = {edge: label for edge, label in nx.get_edge_attributes(G_min, 'label').items() if label != ' '}
    nx.draw(G_min, pos_min, with_labels=True, node_size=800, node_color='lightblue')
    nx.draw_networkx_edge_labels(G_min, pos_min, edge_labels=labels_min)
    plt.title("Minimized DFA Visualization")
    plt.figure(figsize=(30, 16))  # Ajusta el tamaño de la figura (ancho x alto) según tus preferencias
    plt.show()


    symbols = set()
    for _, _, label in min_dfa.edges(data='label'):
            symbols.add(label)

    with open('descripcion_afd_minimizado.txt', 'w') as file:
            file.write("ESTADOS = {}\n".format(sorted(min_dfa.nodes)))
            file.write("SIMBOLOS = {}\n".format(sorted(symbols)))
            file.write("INICIO = {}\n".format(min_dfa.graph['start']))
            file.write("ACEPTACION = {}\n".format(sorted(min_dfa.graph['accept'])))
            file.write("TRANSICIONES = {}\n".format(sorted(min_dfa.edges(data='label'))))

    #SIMULACION DEL AFD
    result = check_membership(min_dfa, w)

    if result:
        print(f"'{w}' belongs to L({regex})")
    else:
        print(f"'{w}' does not belong to L({regex})")


#Algoritmo para minimizar un AFD hecho por construcción directa.


