import numpy as np
import pandas as pd
import random

# Definir los parámetros del algoritmo genético
population_size = 500
chromosome_length = 10
num_generations = 1000
mutation_rate = 0.01

#Read kro file
with open('kroA100.tsp', 'r') as file:
    data = file.readlines()

df = pd.DataFrame([line.split() for line in data], columns=['No.','x', 'y'])
coords = df.drop(columns = 'No.')
coords_list = coords.astype(int).values.tolist() #Float list

#print(coords_list)

cantidad_generaciones = 1000
poblacion = 500

def permute_list(coords_list):
    permuted_list = random.sample(coords_list, len(coords_list))

# Generar población aleatoria
population_size = 500
num_cities = len(coords_list)

# Generar una solución aleatoria única sin elementos repetidos
unique_solutions = []
while len(unique_solutions) < population_size:  # Generar 500 soluciones únicas
    solution = np.random.permutation(num_cities)
    if not any(np.array_equal(solution, s) for s in unique_solutions):
        unique_solutions.append(solution)


print('POBLACION')
print(unique_solutions)


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
