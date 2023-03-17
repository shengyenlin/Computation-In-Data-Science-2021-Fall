import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import os
import json
import gc

def print_solution_route(algo_name, record, sol, store_names):
    print(f"{algo_name} result:")
    print(f"The best route length is {round(record[-1], 3)}")
    print(f"Best route:")
    for i in range(len(sol)):
        if i != len(sol) - 1:
            print(store_names[sol[i]], end = "->")
        else:
            print(store_names[sol[i]])
    print("")

def read_distance_matrix(path):
    if os.path.exists(path):
        with open(file=path, mode='r', encoding='utf-8') as reader:
            adjacency_matrix = json.load(reader)
    distance_df = pd.DataFrame.from_dict(adjacency_matrix)
    distance_matrix = distance_df.to_numpy()
    store_names = distance_df.columns.to_list()
    return distance_matrix, store_names

class TravelingSalesManProblem:
    def __init__(self, distance_matrix):
        self.distance_matrix = distance_matrix

    def get_uniform_dist(self):
        """Get probability from a uniform distribution (0, 1)
        """
        return round(np.random.rand(), 2)

    def generate_random_sol(self):
        sol = np.random.choice(
            np.arange(0, len(self.distance_matrix)), size = len(self.distance_matrix),
            replace = False)
        return list(sol)

    def get_route_length(self, sol):
        ttl_length = 0
        for i in range(len(sol) - 1):
            ttl_length += self.distance_matrix[sol[i]][sol[i+1]]
        #add the distance from the last store to the starting store
        ttl_length += self.distance_matrix[sol[-1]][sol[0]]
        return ttl_length

    def get_neighbors(self, sol):
        """Generate all possible neighbor solutions by swapping elements in the original solution
        """
        neighbors = []
        for i in range(len(sol)):
            for j in range(i+1, len(sol)):
                neighbor = sol.copy()
                neighbor[i] = sol[j]
                neighbor[j] = sol[i]
                neighbors.append(neighbor)
        return neighbors

    def get_best_neighbor(self, neighbors):
        best_neighbor_route_length = self.get_route_length(neighbors[0])
        best_neighbor = neighbors[0]
        for neighbor in neighbors:
            current_route_length = self.get_route_length(neighbor)
            if current_route_length < best_neighbor_route_length:
                best_neighbor_route_length = current_route_length
                best_neighbor = neighbor
        return best_neighbor, best_neighbor_route_length


    def initialize_population(self, pop_size):
        population = []
        for i in range(0, pop_size):
            population.append(self.generate_random_sol())
        return population

class HillClimbing(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None

    def solve(self, verbose = False):
        #Initialize a solution
        best_solution = self.generate_random_sol()
        best_route_length = self.get_route_length(best_solution)
        count = 0
        #Loop until it's not updating
        while count < self.max_iter:
            neighbors = self.get_neighbors(best_solution)
            best_neighbor, best_neighbor_route_length = self.get_best_neighbor(neighbors)
            if best_neighbor_route_length < best_route_length:
                best_solution = best_neighbor
                best_route_length = best_neighbor_route_length
                self.best_route = best_solution
            else:
                break
            self.best_length_record.append(best_route_length)
            count += 1
            if verbose:
                print(f"Iteration {count}, best length is {round(best_route_length, 3)}")
        self.best_route = best_solution

class RandomWalk(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None

    def solve(self, verbose):
        #Initialize a solution
        current_solution = self.generate_random_sol()
        current_route_length = self.get_route_length(current_solution)
        best_solution = current_solution
        best_route_length = current_route_length
        #Start solving
        count = 0
        update_threshold = 0.5
        while count < self.max_iter:
            neighbors = self.get_neighbors(current_solution)
            best_neighbor, best_neighbor_route_length = self.get_best_neighbor(neighbors)
            #updating best solution
            if best_neighbor_route_length < current_route_length:
                best_solution = best_neighbor
                best_route_length = best_neighbor_route_length

            #Must Update
            current_solution = best_neighbor
            current_route_length = best_neighbor_route_length
            count += 1
            self.best_length_record.append(best_route_length)
            if verbose:
                print(f"Iteration {count}, best length is {round(best_route_length, 3)}")
        self.best_route = best_solution

class SimulatedAnnealing(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None

    def solve(self, init_temp, stop_temp, max_iter, max_patience, eplison, verbose):
        count = 0 #Number of iteration
        patience = 0

        #Initialize a solution
        cur_solution = self.generate_random_sol()
        cur_route_length = self.get_route_length(cur_solution)
        best_solution = cur_solution
        best_route_length = cur_route_length
        cur_temp = self.get_temp(eplison, count, init_temp)
        
        #Stopping condition
        #When the objective value did't update for `max_patience` times OR
        #The temperature is lower than `temp_stop` OR
        #number of iterations is larger than 100
        while count < max_iter and patience < max_patience and cur_temp > stop_temp:
            update_threshold = self.get_uniform_dist()
            count+=1
            cur_temp = self.get_temp(eplison, count, init_temp)
            #Generating best neighbor solutions
            neighbors = self.get_neighbors(best_solution)
            best_neighbor, best_neighbor_route_length = self.get_best_neighbor(neighbors)
            #Calculate the difference between current (global) best solution
            #with best neighbor solution (local)
            diff = best_neighbor_route_length - cur_route_length
            update_prob = self.get_boltzmann_prob(diff, cur_temp)
            if update_prob > update_threshold:
                if best_neighbor_route_length < best_route_length:
                    best_solution = best_neighbor
                    best_route_length = best_neighbor_route_length
                #update
                cur_solution = best_neighbor
                cur_route_length = best_neighbor_route_length
                patience = 0
            else:
                patience += 1          
            self.best_length_record.append(best_route_length)
            if verbose:
                print(f"Iteration {count}", f", best length is {round(best_route_length, 2)}")
        self.best_route = best_solution

    def get_boltzmann_prob(self, delta_f, temperature):
        prob = min(1, np.exp(- delta_f / temperature))
        return prob

    def get_temp(self, eplison, num_iter, initial_temp):
        temp = ((1 - eplison) ** num_iter) * initial_temp
        return temp

class GeneticAlgorithm(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None

    def solve(self, pop_size, max_iter, mutation_rate, verbose):
        count = 0
        global_best_length = 0
        global_best_candidate = None

        population = self.initialize_population(pop_size)
        population_length = list(map(self.get_route_length, population))
        best_length = min(population_length)
        global_best_length = best_length
        global_best_candidate = population[population_length.index(global_best_length)]
        
        while count < max_iter:
            if verbose:
                print(f"Iteration {count + 1}, current generation best = {round(best_length, 3)}, global best {round(global_best_length, 3)}")
            mating_pool = self.roulette_wheel_selection(population)
            #print(mating_pool)
            offsprings = self.breed(mating_pool)
            population = self.mutate_population(offsprings, mutation_rate)
            
            #update best candidate
            population_length = list(map(self.get_route_length, population))
            best_length = min(population_length)
            if best_length < global_best_length:
                global_best_candidate = population[population_length.index(best_length)]  
                global_best_length = best_length          
                #best_items = np.array(self.items)[global_best_candidate]
            
            self.best_length_record.append(global_best_length)
            count +=1
        self.best_route = global_best_candidate

    #直接變成mating pool，不用分成a, b
    def roulette_wheel_selection(self, population): 
        population_fitness = sum([self.get_route_length(x) for x in population])
        #when length is higher, the chosen probability show be lower
        chromosome_probabilities = [1/(self.get_route_length(x)/population_fitness) for x in population]
        #use softmax to change it to legal probability defition
        chromosome_probabilities = self.softmax(chromosome_probabilities)
        mating_pool = []
        for i in range(len(population)):
            idx = np.random.choice(list(range(len(population))), p = chromosome_probabilities)
            mating_pool.append(population[idx])
        return mating_pool

    def softmax(self, x):
        f_x = np.exp(x) / np.sum(np.exp(x))
        return f_x

    #對兩個parent (from mating pool)做cross-over
    #ordered cross-over
    def cross_over(self, parent_a, parent_b):
        child = []
        child_part1 = []
        child_part2 = []
        
        gene_a = int(np.random.rand() * len(parent_a))
        gene_b = int(np.random.rand() * len(parent_a))
        
        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)

        for i in range(start_gene, end_gene):
            child_part1.append(parent_a[i])
            
        child_part2 = [item for item in parent_b if item not in child_part1]

        child = child_part1 + child_part2
        return child

    #隨便從mating pool裡面挑兩個出來cross over
    def breed(self, mating_pool):
        children = []
        for i in range(0, len(mating_pool)):
            child = self.cross_over(mating_pool[i], mating_pool[len(mating_pool)-i-1])
            children.append(child)
        return children

    def mutate(self, children, mutation_rate):
        for i in range(len(children)):
            if (np.random.rand() < mutation_rate):
                swap = int(np.random.rand() * len(children))
                city1 = children[i]
                city2 = children[swap]
                children[i] = city2
                children[swap] = city1
        return children

    def mutate_population(self, population, mutation_rate):
        mutated_pop = []
        for i in range(len(population)):
            mutate_child = self.mutate(population[i], mutation_rate)
            mutated_pop.append(mutate_child)
        return mutated_pop

class TabuSearch(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None

    def solve(self, max_tabu_list_length, verbose):
        #Initialize a solution
        current_solution = self.generate_random_sol()
        current_route_length = self.get_route_length(current_solution)
        best_solution = current_solution
        best_route_length = current_route_length
        count = 0 
        tabu_list = []

        #Start solving
        while count < self.max_iter:
            #print(current_solution)
            neighbors = self.get_neighbors(current_solution)
            best_neighbor = None
            best_neighbor_route_length = 10 ** 9
            for neighbor in neighbors:
                neighbor_route_length = self.get_route_length(neighbor)

                update = neighbor not in tabu_list and \
                            neighbor_route_length < best_neighbor_route_length

                if update:
                    best_neighbor = neighbor
                    best_neighbor_route_length = neighbor_route_length

            current_solution = best_neighbor
            current_route_length = best_neighbor_route_length
            if current_solution is not None:
                if current_route_length <= best_route_length:
                    best_route_length = current_route_length
                    best_solution = current_solution

                #even if best neighbor is larger than best solution, we still append it to tabu list    
                tabu_list.append(current_solution)

                if len(tabu_list) > max_tabu_list_length:
                    tabu_list = tabu_list[:max_tabu_list_length]

                #aspiration criteria
                tabu_list_length = list(map(self.get_route_length, tabu_list))
                if max(tabu_list_length) >= current_route_length:
                    current_solution = tabu_list[tabu_list_length.index(max(tabu_list_length))]
                tabu_list.remove(current_solution)

            count += 1
            self.best_length_record.append(best_route_length)
            self.best_route = best_solution
            if verbose:
                print(f"Iteration {count}, best length is {round(best_route_length, 3)}")


class ParticalSwarmOptimization(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None
    
    def solve(self, max_iter, pop_size, verbose):
        #initialization
        num_iter = 0
        population_current, global_best = self.initialize_PSO(pop_size)
        population_current_length = list(map(self.get_route_length, population_current))
        local_best = population_current #each particle has its local best route
        local_best_length = list(map(self.get_route_length, local_best))
        global_best_length = max(local_best_length)

        velocity = local_best #use local best candidate as velocity initially

        #parameter settings
        c1 = c2 = w = 1
        while num_iter < max_iter:
            #cross over local and current
            route_cross_local_current = []
            route_cross_global_current = []
            route_cross_local_global = []
            velocity_next = []
            population_next = []
            for i in range(pop_size):
                route_cross_local_current.append(self.ordered_random_cross_over(local_best[i], population_current[i]))
            #cross over global and current
            for i in range(pop_size):
                route_cross_global_current.append(self.ordered_random_cross_over(global_best, population_current[i]))
            #cross over the two above
            for i in range(pop_size):
                route_cross_local_global.append(self.ordered_random_cross_over(route_cross_local_current[i], route_cross_local_current[i])) 
            #cross over velocity and the above result
            for i in range(pop_size):
                velocity_next.append(self.ordered_random_cross_over(velocity[i], route_cross_local_global[i]))
            velocity = velocity_next
            #generate next generation
            for i in range(pop_size):
                population_next.append(self.ordered_random_cross_over(velocity_next[i], population_current[i]))

            #update local best
            population_next_length = list(map(self.get_route_length, population_next))
            for i in range(pop_size):
                if population_next_length[i] > population_current_length[i]:
                    local_best[i] = population_next[i]
                    local_best_length[i] = population_next_length[i] 
                else:
                    local_best[i] = population_current[i]
                    local_best_length[i] = population_current_length[i]

            #update global best
            if min(local_best_length) < global_best_length:
                global_best_length = min(local_best_length)
                global_best = local_best[local_best_length.index(global_best_length)]
            
            num_iter += 1
            self.best_length_record.append(global_best_length)
            if verbose:
                print(f"Iteration {num_iter}, global best length is {round(global_best_length, 3)}")
            self.best_route = global_best

    def generate_two_rv(self, isSorted):
        r1 = np.random.rand()
        r2 = np.random.rand()
        if isSorted:
            return min(r1, r2), max(r1, r2)
        else:
            return r1, r2

    def ordered_random_cross_over(self, route_best, route_current):
        #print(route_best)
        return_route = []
        route_part1 = []
        route_part2 = []
        r1, r2 = self.generate_two_rv(isSorted=True)
        start_route = int(r1 * len(route_current))
        end_route = int(r2 * len(route_current))
        for i in range(start_route, end_route):
            route_part1.append(route_best[i])
        route_part2 = [store for store in route_current if store not in route_part1]
        return_route = route_part1 + route_part2
        return return_route

    def initialize_PSO(self, pop_size):
        population = self.initialize_population(pop_size)
        population_length = list(map(self.get_route_length, population))
        best_length = min(population_length)
        best_candidate = population[population_length.index(best_length)]
        return population, best_candidate #use best candidate in initialization as velocity

class AntColonyOptimization(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None

    def solve(self, max_iter, num_ants, Q, rho , alpha, beta, verbose):
        pheromone_scaling_factor = 2
        current_sol, visibility, pheromone_map = self.initialize_ACO(num_ants, Q)
        fitness = self.compute_fitness(pheromone_map, visibility, alpha, beta)
        global_best_length = 1e+5
        global_best_route = None
        num_iter = 0
        while num_iter < max_iter:
            current_sol = self.construct_solution(num_ants, current_sol, fitness)
            current_route_length = list(map(self.get_route_length, current_sol))
            local_best_length = max(current_route_length)
            local_best_route = current_sol[current_route_length.index(local_best_length)]
            local_worst_length = min(current_route_length)
            drop_pheromone = pheromone_scaling_factor * local_best_length / local_worst_length
            if local_best_length < global_best_length:
                global_best_length = local_best_length
                global_best_route = local_best_route
            
            #update pheromone and fitness
            pheromone_map = self.update_pheromone(pheromone_map, rho, local_best_route, drop_pheromone)
            fitness = self.compute_fitness(pheromone_map, visibility, alpha, beta)
            self.best_length_record.append(global_best_length)
            if verbose:
                print(f"Iteration {num_iter + 1}, local best length is {round(local_best_length, 3)}, global best length is {round(global_best_length, 3)}")
            num_iter += 1 
        self.best_route = global_best_route
    
    def update_pheromone(self, pheromone_map, rho, local_best_route, drop_pheromone):
        #update the pheromone on the best path
        pheromone_map = pheromone_map * (1-rho)
        num_city = len(local_best_route)
        for i in range(num_city - 1):
            pheromone_map[local_best_route[i]][local_best_route[i+1]] += drop_pheromone
        pheromone_map[local_best_route[num_city - 1]][local_best_route[0]] += drop_pheromone
        return pheromone_map

    def initialize_ACO(self, num_ants, Q):
        pop = np.zeros(
            (num_ants, len(self.distance_matrix)), dtype = int
        )
        eps = np.identity(len(self.distance_matrix)) * 1e+8
        visibility = 1 / np.array(self.distance_matrix + eps) #prevent divide by zero
        pheromone_map = np.full(
            (len(self.distance_matrix), len(self.distance_matrix))
            , Q)
        return pop, visibility, pheromone_map

    def construct_solution(self, num_ants, pop, fitness):
        #assign random city as starting point for each ant
        for i in range(num_ants):
            pop[i][0] = np.random.randint(num_ants)
            pop = self.construct_solution_for_ant_i(i, fitness, pop)
        return pop
        
    def do_roulette_wheel_ACO(self, fitness, current_city):       
        next_city_fitness = fitness[current_city]
        transition_probability = next_city_fitness / sum(next_city_fitness)
        rand = np.random.rand()
        sum_prob = 0
        for (i, prob) in enumerate(transition_probability):
            sum_prob += prob
            if sum_prob >= rand:
                return i

    def construct_solution_for_ant_i(self, i, fitness, pop):
        fitness_ = fitness.copy()
        city_candidate = list(np.arange(len(fitness)))
        current_city = pop[i][0]
        #mask the fitness of chosen city
        fitness_[:, current_city] = np.zeros(len(fitness))
        city_candidate.remove(current_city)
        for j in range(1, len(fitness)):
            next_city = self.do_roulette_wheel_ACO(fitness_, current_city)
            #print(next_city)
            pop[i][j] = next_city
            fitness_[:, next_city] = np.zeros(len(fitness))
            current_city = next_city
        return pop

    def compute_fitness(self, pheromone_map, visibility, alpha, beta):
        return (pheromone_map ** alpha) * (visibility ** beta)

class CuckooSearch(TravelingSalesManProblem):
    def __init__(self, distance_matrix, max_iter):
        super().__init__(distance_matrix)
        self.max_iter = max_iter
        self.best_length_record = []
        self.best_route = None

    def levy_flight(self, u):
            return np.power(u,-1.0/3.0)

    def randF(self):
        return np.random.uniform(0.0001,0.9999)

    def calculate_distance(self, path):
        index = path[0]
        distance = 0
        for nextIndex in path[1:]:
            distance += self.distance_matrix[index][nextIndex]
            index = nextIndex
        return distance+self.distance_matrix[path[-1]][path[0]]

    def swap(self, sequence,i,j):
        temp = sequence[i]
        sequence[i]=sequence[j]
        sequence[j]=temp

    def two_opt_move(self,nest,a,c):
        nest = nest[0][:]
        self.swap(nest,a,c)
        return (nest,self.calculate_distance(nest))
        

    def double_bridge_move(self,nest,a,b,c,d):
        nest = nest[0][:]
        self.swap(nest,a,b)
        self.swap(nest,b,d)
        return (nest,self.calculate_distance(nest))

    def hill_climbing(self,sequence):
        improvements = True
        n = len(self.distance_matrix)
        path = sequence[0][:]
        bestPath = path[:]
        distance = self.calculate_distance(path)
        while(improvements):
            improvements = False;
            for i in range(n-1):
                self.swap(path,i,i+1)
                newDistance = self.calculate_distance(path);
                if(distance>newDistance):
                    improvements = True
                    bestPath = path[:]
                    distance = newDistance;
                else:
                    self.swap(path,i,i+1)
        return (bestPath,self.calculate_distance(bestPath))
    
    def solve(self, num_nests):
        pa = int(0.2*num_nests)
        pc = int(0.6*num_nests)

        n = len(self.distance_matrix)

        nests = []

        init_path=np.arange(0,n).tolist()
        index = 0
        for i in range(num_nests):
            if index == n-1:
                index = 0
            self.swap(init_path,index,index+1)
            index+=1
            nests.append((init_path[:],self.calculate_distance(init_path)))

        nests.sort(key=lambda tup: tup[1])

        for t in range(self.max_iter):
            cuckooNest = nests[np.random.randint(0,pc)]
            if(self.levy_flight(self.randF())>2):
                if(np.random.randint(0,1)==1):
                    cuckooNest = self.double_bridge_move(
                        cuckooNest,np.random.randint(0,n-1),np.random.randint(0,n-1),
                        np.random.randint(0,n-1),np.random.randint(0,n-1))
                else:
                    cuckooNest = self.hill_climbing(cuckooNest)
            else:
                cuckooNest = self.two_opt_move(cuckooNest,np.random.randint(0,n-1),np.random.randint(0,n-1))
            randomNestIndex = np.random.randint(0,num_nests-1)
            if(nests[randomNestIndex][1]>cuckooNest[1]):
                nests[randomNestIndex] = cuckooNest
            for i in range(num_nests-pa,num_nests):
                nests[i] = self.two_opt_move(nests[i],np.random.randint(0,n-1),np.random.randint(0,n-1))
            nests.sort(key=lambda tup: tup[1])
            self.best_length_record.append(nests[0][1])
        self.best_route = nests[0][0]