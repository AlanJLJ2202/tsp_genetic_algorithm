import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

generations = 1000
population_size = 100
k = 2

with open('kroA100.tsp', 'r') as file:
    data = file.readlines()

#Nota: Tal vez es mejor no quitar la columna de "No." por que contiene el indice de la ciudad
cities_data = pd.DataFrame([line.split() for line in data], columns=['No.','x', 'y'])
cities = cities_data.astype(int).values.tolist()

#label_cities = [city for city in cities]
label_cities = []

for city in cities:
    label_cities.append(city[0])

#print(label_cities)

labeled_cities = {}

for city in cities:
    labeled_cities[city[0]] = [city[1], city[2]]

#print(labeled_cities)

'''
def generate_population(label_cities, population_size):
    unique_solutions = []
    while len(unique_solutions) < population_size:  # Generar 500 soluciones únicas
        solution = np.random.permutation(label_cities)
        if not any(np.array_equal(solution, s) for s in unique_solutions):
            unique_solutions.append(solution)
    #unique_solutions = [tuple(elemento) for elemento in unique_solutions]
    return unique_solutions
'''

def fitness(solution):
    euclidean_distances = []
    
    for i in range(len(solution)):
        x1, y1 = labeled_cities[solution[i]]


        if i == (len(solution)-1):
            x2,y2 = labeled_cities[solution[0]]
            distance = euclidean_distance(x1, y1, x2, y2)
            euclidean_distances.append(distance)
        else:
            x2, y2 = labeled_cities[solution[i+1]]
            distance = euclidean_distance(x1, y1, x2, y2)
            euclidean_distances.append(distance)
    return sum(euclidean_distances)

def euclidean_distance(x1, y1, x2, y2):
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    distance = int(distance + 0.5)
    return distance

def evaluate_solutions(solutions):
    evaluated_solutions = []
    for solution in solutions:
        aptitud = fitness(solution)
        evaluated_solution = list(solution)
        evaluated_solution.insert(0, aptitud)
        evaluated_solutions.append(evaluated_solution)
    return evaluated_solutions

def tournament(population, k):
    selected = []

    for i in range(2):
        tournament = random.sample(population, k)
        winner = min(tournament, key=lambda x: x[0])
        selected.append(winner)
    
    return selected

def PMX(solution1, solution2):
    child = []
    size = len(solution1)
    cut1, cut2 = random.sample(range(len(solution1)), 2)
    
    for i in range(size):
        child.append(None)
    
    if cut1 > cut2:
        cut1,cut2 = cut2,cut1
    
    child[cut1:cut2+1] = solution1[cut1:cut2+1]
    #print("child" , child)
    parent2 = solution2[cut1:cut2+1]
    #print("parent2", parent2)

    for i in range(len(child)):
        for j in range(len(parent2)):
            if parent2[j] not in child:
                idxchild = solution2.index(solution1[solution2.index(parent2[j])])
                #print(child)
                while child[idxchild] != None:
                    idxchild = solution2.index(solution1[idxchild])
                child[idxchild] = parent2[j]
        
        if child[i] == None:
            child[i] = solution2[i]
            
    return child


def mutation(solution):
    city1, city2 = random.sample(range(len(solution)), 2)
    solution[city1], solution[city2] = solution[city2], solution[city1]
    return solution


def generate_population(label_cities, population_size):
    
    #random.shuffle(label_cities)

    solutions = []
    for i in range(population_size):
        permutation = random.sample(label_cities, len(label_cities))
        if permutation not in solutions:  
            solutions.append(permutation)
    return solutions

parent1 = [41, 7, 89, 45, 53, 22, 59, 51, 19, 23, 33, 46, 56, 78, 93, 90, 9, 6, 44, 48, 15, 2, 49, 94, 71, 39, 65, 73, 64, 84, 21, 100, 43, 18, 67, 58, 50, 69, 66, 85, 12, 95, 77, 42, 97, 26, 1, 92, 99, 16, 34, 25, 72, 3, 37, 76, 74, 47, 11, 55, 31, 4, 83, 8, 60, 87, 82, 10, 80, 29, 35, 14, 32, 57, 62, 63, 54, 96, 79, 24, 30, 88, 17, 28, 68, 75, 98, 5, 91, 36, 52, 70, 40, 86, 20, 27, 38, 61, 81, 13]
parent2 = [165313, 14, 3, 100, 56, 10, 48, 55, 20, 74, 71, 86, 78, 69, 32, 79, 27, 11, 97, 93, 90, 67, 44, 64, 60, 98, 92, 77, 52, 95, 29, 2, 30, 15, 66, 89, 22, 26, 47, 70, 57, 81, 91, 94, 18, 54, 17, 4, 73, 46, 1, 40, 43, 16, 88, 39, 84, 50, 8, 41, 76, 33, 83, 24, 82, 38, 25, 59, 23, 87, 96, 5, 19, 21, 36, 80, 65, 9, 68, 6, 62, 45, 85, 75, 37, 99, 31, 61, 72, 49, 34, 51, 42, 7, 63, 58, 12, 13, 35, 28, 53]

population = generate_population(label_cities, population_size)
best_solutions = []
percentage = 0
for i in range(generations):
    new_percentaje = int((i * 100)/generations)
    if percentage != new_percentaje:
        percentage = new_percentaje
        print(percentage, "%")
    
    evaluated_solutions = evaluate_solutions(population)
    best = min(evaluated_solutions, key=lambda x: x[0])
    best_solutions.append(best[0])
    
    new_generation = []
    while len(new_generation) < len(population):
        parents = tournament(evaluated_solutions, 2)
        child = PMX(parents[0][1:], parents[1][1:])
        #mutated_child = mutation(child) #Checar este por que cuando lo descomento se vuelve bien loca la grafica
        new_generation.append(child)
    
    population = new_generation
print('Mejor solucion encontrada')
print('----------------------------------------')
print(best)
print('Lista de mejores soluciones')
print('----------------------------------------')
print(best_solutions)
plt.plot(best_solutions)
plt.show()

#population = generate_population(label_cities)
#print(population)
#print("Evaluated", evaluated_population)