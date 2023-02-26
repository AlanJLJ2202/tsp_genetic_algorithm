import numpy as np
import pandas as pd
import random

# Definir los parámetros del algoritmo genético
chromosome_length = 10
mutation_rate = 0.01
cantidad_generaciones = 1000
poblacion = 500


#Read kro file
with open('kroA100.tsp', 'r') as file:
    data = file.readlines()

#Nota: Tal vez es mejor no quitar la columna de "No." por que contiene el indice de la ciudad
df = pd.DataFrame([line.split() for line in data], columns=['No.','x', 'y'])
coords = df.drop(columns = 'No.')
coords_list = coords.astype(int).values.tolist() #Int list

#Formula de la distancia euclidiana
def euclidean_distance(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def permute_list(coords_list):
    permuted_list = random.sample(coords_list, len(coords_list))

# Generar población aleatoria
population_size = 500
num_cities = len(coords_list)

#---Esto a lo mejor se puede optimizar
# Generar una solución aleatoria única sin elementos repetidos
unique_solutions = []
while len(unique_solutions) < population_size:  # Generar 500 soluciones únicas
    solution = np.random.permutation(coords_list)
    if not any(np.array_equal(solution, s) for s in unique_solutions):
        unique_solutions.append(solution)


#Calcula la aptitud de la solucion, este metodo simplemente recibe una solucion y le calcula
#su distancia euclidiana, entre mas distancia menos apta es la solucion
def fitness(solution):
    euclidean_distances = []
    for i in range(len(solution)):
        x1 = solution[i][0]
        y1 = solution[i][1]
        
        if i == (len(solution)-1):
            x2 = solution[0][0]
            y2 = solution[0][1]
            distance = euclidean_distance(x1, y1, x2, y2)
            euclidean_distances.append(distance)
        else:
            x2 = solution[i+1][0]
            y2 = solution[i+1][1]
            distance = euclidean_distance(x1, y1, x2, y2)
            euclidean_distances.append(distance)
        
    return sum(euclidean_distances)

#Evalua cada una de las soluciones, es decir calcula la distancia de cada una de ellas
#y la agrega al inicio del array de la solucion, esto sirve para el metodo de seleccion
def evaluate_solutions(unique_solutions):
    evaluated_solutions = []
    for solution in unique_solutions:
        aptitud = fitness(solution)
        evaluated_solution = list(solution)
        evaluated_solution.insert(0, aptitud)
        print("--------")
        print(evaluated_solution)
        evaluated_solutions.append(evaluated_solution)
    return evaluated_solutions

#Aqui solo lo puse para ejecutarlo xd
evaluate_solutions(unique_solutions)

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
