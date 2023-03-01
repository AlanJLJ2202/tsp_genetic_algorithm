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
    print(selected)
    return selected


#Metodo para mutar un elemento, este metodo recibe una solucion de ciudades 
#e intercambia dos de estas de manera aleatoria
def mutation(solution):
    city1, city2 = random.sample(range(len(solution)), 2)
    solution[city1], solution[city2] = solution[city2], solution[city1]
    print(solution)
    return solution

#Metodo para realizar el cruce de dos soluciones, este metodo recibe dos soluciones
#Metodo de cruza PMX
def crossover(solution1, solution2):
    # Seleccionar dos puntos aleatorios
    point1, point2 = sorted(random.sample(range(len(solution1)), 2))
    # Intercambiar los valores entre los dos puntos
    solution1[point1:point2], solution2[point1:point2] = solution2[point1:point2], solution1[point1:point2].copy()
    # Corregir los valores que se repiten
    for solution in (solution1, solution2):
        for point in (point1, point2):
            while int(solution[point]) in [int(x) for x in solution[:point] + solution[point+1:]]:
                index = solution.index(solution[point])
                if index == point:
                    break
                solution[index] = solution[point]

    return solution1, solution2

#Por cada hijo se va a realizar la mutacion

#Aqui solo lo puse para ejecutarlo xd
population = generate_population(cities)
evaluated_population = evaluate_solutions(population)
print('POBLACION EVALUADA')
print(evaluated_population)
print('---------------------------------')
torneos = tournament(evaluated_population, k)
crossover(torneos[0], torneos[1])


#mutation(population[0])
