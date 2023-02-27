import numpy as np
import pandas as pd
import random

# Definir los parámetros del algoritmo genético
chromosome_length = 10
mutation_rate = 0.01
cantidad_generaciones = 1000
population_size = 500
k = 2

#Read kro file
with open('kroA100.tsp', 'r') as file:
    data = file.readlines()

#Nota: Tal vez es mejor no quitar la columna de "No." por que contiene el indice de la ciudad
cities_data = pd.DataFrame([line.split() for line in data], columns=['No.','x', 'y'])
cities = cities_data.astype(int).values.tolist()

#Formula de la distancia euclidiana
def euclidean_distance(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

# Generar una solución aleatoria única sin elementos repetidos
def generate_population(cities):
    unique_solutions = []
    while len(unique_solutions) < population_size:  # Generar 500 soluciones únicas
        solution = np.random.permutation(cities)
        if not any(np.array_equal(solution, s) for s in unique_solutions):
            unique_solutions.append(solution)
    unique_solutions
    return unique_solutions

#Calcula la aptitud de la solucion, este metodo simplemente recibe una solucion y le calcula
#su distancia euclidiana, entre mas distancia menos apta es la solucion 
#x = solution[i][1]
#y = solution[i][2]
def fitness(solution):
    euclidean_distances = []
    
    for i in range(len(solution)):
        x1 = solution[i][1]
        y1 = solution[i][2]
        
        if i == (len(solution)-1):
            x2 = solution[0][1]
            y2 = solution[0][2]
            distance = euclidean_distance(x1, y1, x2, y2)
            euclidean_distances.append(distance)
        else:
            x2 = solution[i+1][1]
            y2 = solution[i+1][2]
            distance = euclidean_distance(x1, y1, x2, y2)
            euclidean_distances.append(distance)
    
    return sum(euclidean_distances)

#Evalua cada una de las soluciones, es decir calcula la distancia de cada una de ellas
#y la agrega al inicio del array de la solucion, esto sirve para el metodo de seleccion
def evaluate_solutions(solutions):
    evaluated_solutions = []
    for solution in solutions:
        aptitud = fitness(solution)
        evaluated_solution = list(solution)
        evaluated_solution.insert(0, aptitud)
        evaluated_solutions.append(evaluated_solution)
    return evaluated_solutions

#Metodo para realizar la compentencia entre k elementos, se usa la funcion min para seleccionar
#la solucion con la menor distancia obtenida, este metodo retorna los mejores individuos 
#de la poblacion
def tournament(population, k):
    selected = []

    while len(selected) < len(population):
        tournament = random.sample(population, k)
        winner = min(tournament, key=lambda x: x[0])
        selected.append(winner)
    return selected


#Metodo para mutar un elemento, este metodo recibe una solucion de ciudades 
#e intercambia dos de estas de manera aleatoria
def mutation(solution):
    city1, city2 = random.sample(range(len(solution)), 2)
    solution[city1], solution[city2] = solution[city2], solution[city1]
    return solution


#Aqui solo lo puse para ejecutarlo xd
population = generate_population(cities)
evaluated_population = evaluate_solutions(population)
tournament(evaluated_population, k)

'''
# Definir la función de aptitud (para maximizar la suma de los elementos del cromosoma)
def fitness_function(chromosome):
    return np.sum(chromosome)

# Evaluar la aptitud de la población inicial
fitness_scores = np.apply_along_axis(fitness_function, 1, population)

# Repetir durante un número determinado de generaciones
for generation in range(num_generations):

    # Selección
    selection_probabilities = fitness_scores / np.sum(fitness_scores)
    selected_indices = np.random.choice(population_size, size=population_size, p=selection_probabilities)
    selected_population = population[selected_indices]

    # Cruce
    offspring_population = np.zeros_like(selected_population)
    for i in range(population_size):
        parent1 = selected_population[i]
        parent2 = selected_population[(i+1)%population_size]
        swap_indices = np.random.choice(chromosome_length, size=2, replace=False)
        offspring = np.copy(parent1)
        offspring[swap_indices[0]] = parent2[swap_indices[0]]
        offspring[swap_indices[1]] = parent2[swap_indices[1]]
        offspring_population[i] = offspring

    # Mutación
    mutation_mask = np.random.random(offspring_population.shape) < mutation_rate
    mutation_values = np.random.randint(2, size=offspring_population.shape)
    offspring_population[mutation_mask] = mutation_values[mutation_mask]

    # Evaluar la aptitud de la nueva población
    offspring_fitness_scores = np.apply_along_axis(fitness_function, 1, offspring_population)

    # Reemplazar la población anterior con la nueva población
    population = offspring_population
    fitness_scores = offspring_fitness_scores

# Encontrar el mejor cromosoma de la última generación
best_chromosome_index = np.argmax(fitness_scores)
best_chromosome = population[best_chromosome_index]

print("Mejor solución encontrada:", best_chromosome)
print("Valor de aptitud:", fitness_function(best_chromosome))
'''
