import math
import numpy as np
import random

points = np.array([[[1,1],[4,1],[2,1],[3,1]],[[1,1],[2,1],[3,1],[4,1]],[[2,1],[1,1],[3,1],[4,1]]])
iterations = 10
pop_count = 3
chromosome_size = 4
pop_size = (pop_count,chromosome_size,2)
offspring_count = 3
parents_size = 2
offspring_size = (offspring_count,chromosome_size,2)  

def euclidean_distance(x1,x2):
    dist = math.sqrt( ((x1[0]-x2[0])**2)+((x1[1]-x2[1])**2) )
    return dist

def total_euclidean_distance(points):
    tot_dist_list = 0
    for index,eachpoint in enumerate(points):
        if index < len(points)-1:
            tot_dist_list = tot_dist_list+euclidean_distance(points[index],points[index+1])
    tot_dist_list = tot_dist_list+euclidean_distance(points[0],points[-1])
    return tot_dist_list

def fitness(points):
    fitness_score = []
    for idx in range(len(points)):
        fitness = total_euclidean_distance(points[idx])
        fitness_score.append(fitness)
    return fitness_score
 
def selection(points, fitness, num_parents):
    parents = np.empty((parents_size,points.shape[1],points.shape[2]))
    for parent_num in range(parents_size):
        max_fitness_idx = np.where(fitness == np.min(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num] = points[max_fitness_idx]
        fitness[max_fitness_idx] = 999
    return parents

def crossover(parents, offspring_size):
     offspring = np.empty(offspring_size)
     crossover_point = np.uint8(offspring_size[1]/2)
     for k in range(offspring_size[0]):         
         parent1_idx = k%parents.shape[0]
         parent2_idx = (k+1)%parents.shape[0]
         offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
         child = []
         for each_val in parents[parent2_idx,:].tolist():
             if each_val not in parents[parent1_idx, 0:crossover_point].tolist():
                 child.append(each_val)
         offspring[k, crossover_point:] = np.array(child)
     return offspring

def mutation(offsprings):
    for idx in range(offsprings.shape[0]):
        [idx1,idx2] = random.sample(range(1, 4), 2)
        t = np.copy(offsprings[idx,idx2])
        offsprings[idx,idx2] = offsprings[idx,idx1]
        offsprings[idx,idx1] = t
    return offsprings

def genetic_algo(iterations,points):
    for val in range(iterations):
        fitness_score = fitness(points)
        fitness_score_population = np.copy(fitness_score)
        parents = selection(points, fitness_score,parents_size)
        offsprings = crossover(parents,offspring_size)
        mutation_offsprings = mutation(offsprings)
        fitness_score_offsprings = fitness(mutation_offsprings)
        if(np.min(fitness_score_offsprings) <= np.min(fitness_score_population)):
            points = mutation_offsprings
    idx = np.where(fitness_score_offsprings == np.min(fitness_score_offsprings))
    idx = idx[0][0]
    return mutation_offsprings[idx,:],fitness_score_offsprings[idx]

    
best_path,shortest_dist = genetic_algo(iterations,points)
print(best_path,shortest_dist)
