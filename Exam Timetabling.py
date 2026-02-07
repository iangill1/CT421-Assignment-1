import numpy as np
import random
import matplotlib.pyplot as plt

def read_instance(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    N, K, M = map(int, lines[0].strip().split())
    E = [list(map(int, lines[i].strip().split())) for i in range(1, M + 1)]
    return N, K, M, np.array(E)

def hard_constraints_violations(timetable, E, M, N):
    violations = 0
    for student in range(M):
        slots_seen = set()
        for exam in range(N):
            if E[student][exam] == 1:
                slot = timetable[exam]
                if slot in slots_seen:
                    violations += 1
                else:
                    slots_seen.add(slot)
    return violations

def get_students_slots(student, timetable, E, N):
    return sorted([timetable[exam] for exam in range(N) if E[student][exam] == 1])

def soft_constraints_violations(timetable, E, M, N, K):
    total_penalty = 0
    for student in range(M):
        slots = get_students_slots(student, timetable, E, N)
        if not slots:
            continue
        
        # Consecutive slots penalty
        for i in range(len(slots) - 1):
            if slots[i + 1] - slots[i] == 1:
                total_penalty += 1
        
        # Long exam gap penalty
        if len(slots) >= 2:
            max_gap = max(slots[i + 1] - slots[i] for i in range(len(slots) - 1))
            if max_gap > K // 2:
                total_penalty += 1
        
        # End cluster penalty
        if all(slot >= K - 2 for slot in slots):
            total_penalty += 1
    
    return total_penalty

def evaluate_fitness(timetable, E, M, N, K):
    hard = hard_constraints_violations(timetable, E, M, N)
    soft = soft_constraints_violations(timetable, E, M, N, K)
    return hard * 1000 + soft

def tournament_selection(population, fitnesses, tournament_size=3):
    indices = random.sample(range(len(population)), tournament_size)
    best_idx = min(indices, key=lambda i: fitnesses[i])
    return population[best_idx].copy()

def crossover(parent1, parent2, crossover_rate=0.8):
    if random.random() > crossover_rate:
        return parent1.copy(), parent2.copy()
    point = random.randint(1, len(parent1) - 1)
    return parent1[:point] + parent2[point:], parent2[:point] + parent1[point:]

def mutate(timetable, K, mutation_rate=0.05):
    for i in range(len(timetable)):
        if random.random() < mutation_rate:
            timetable[i] = random.randint(0, K - 1)

def local_search(timetable, E, M, N, K, max_iterations=50):
    """Aggressive hill climbing with full neighborhood exploration"""
    current = timetable.copy()
    current_fitness = evaluate_fitness(current, E, M, N, K)
    
    for _ in range(max_iterations):
        best_move = None
        best_fitness = current_fitness
        
        # Try all possible single-exam moves
        for i in range(N):
            old_slot = current[i]
            for new_slot in range(K):
                if new_slot != old_slot:
                    current[i] = new_slot
                    new_fitness = evaluate_fitness(current, E, M, N, K)
                    if new_fitness < best_fitness:
                        best_fitness = new_fitness
                        best_move = (i, new_slot)
                    current[i] = old_slot
        
        if best_move is None:
            break
        
        # Apply best move
        current[best_move[0]] = best_move[1]
        current_fitness = best_fitness
    
    return current

def run_ga(
    file_path,
    pop_size=100,
    generations=500,
    crossover_rate=0.8,
    mutation_rate=0.05,
    elitism_count=2,
    tournament_size=3,
    seed=None,
):
    if seed is not None:
        random.seed(seed)

    N, K, M, E = read_instance(file_path)
    population = [[random.randint(0, K - 1) for _ in range(N)] for _ in range(pop_size)]
    best_fitness_per_gen = []
    
    stagnation_counter = 0
    adaptive_mutation = mutation_rate

    for gen in range(generations):
        fitnesses = [evaluate_fitness(t, E, M, N, K) for t in population]
        current_best = min(fitnesses)
        best_fitness_per_gen.append(current_best)
        
        # Adaptive mutation: increase if stagnating
        if len(best_fitness_per_gen) > 1 and current_best == best_fitness_per_gen[-2]:
            stagnation_counter += 1
            if stagnation_counter > 20:
                adaptive_mutation = min(0.3, mutation_rate * 2)
        else:
            stagnation_counter = 0
            adaptive_mutation = mutation_rate

        elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:elitism_count]
        new_population = [population[i].copy() for i in elite_indices]
        
        # Apply local search to best solution every 30 generations
        if gen % 30 == 0 and gen > 0:
            best_idx = elite_indices[0]
            new_population[0] = local_search(population[best_idx], E, M, N, K, max_iterations=30)

        while len(new_population) < pop_size:
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            child1, child2 = crossover(parent1, parent2, crossover_rate)
            mutate(child1, K, adaptive_mutation)
            mutate(child2, K, adaptive_mutation)
            new_population.append(child1)
            if len(new_population) < pop_size:
                new_population.append(child2)

        population = new_population

    fitnesses = [evaluate_fitness(t, E, M, N, K) for t in population]
    best_index = fitnesses.index(min(fitnesses))
    best_timetable = population[best_index]
    
    # Final intensive local search refinement
    best_timetable = local_search(best_timetable, E, M, N, K, max_iterations=200)
    best_fitness = evaluate_fitness(best_timetable, E, M, N, K)
    
    return best_timetable, best_fitness_per_gen, E, M, N, K, best_fitness

def find_best_params(file_path):
    param_grid = [
        {"pop_size": 300, "generations": 400, "crossover_rate": 0.9, "mutation_rate": 0.04, "elitism_count": 5, "tournament_size": 5},
        {"pop_size": 400, "generations": 350, "crossover_rate": 0.85, "mutation_rate": 0.06, "elitism_count": 4, "tournament_size": 6},
        {"pop_size": 250, "generations": 450, "crossover_rate": 0.95, "mutation_rate": 0.03, "elitism_count": 3, "tournament_size": 4},
    ]

    restarts = 8
    best = None

    for params in param_grid:
        for _ in range(restarts):
            seed = random.randint(0, 1_000_000)
            result = run_ga(file_path, seed=seed, **params)
            best_fitness = result[-1]
            if best is None or best_fitness < best[0]:
                best = (best_fitness, params, result)

    _, best_params, best_result = best
    return best_params, best_result

def main(): 
    best_params, best_result = find_best_params('tinyexample.txt')
    best_timetable, fitness_per_gen, E, M, N, K, _ = best_result

    hard = hard_constraints_violations(best_timetable, E, M, N)
    soft = soft_constraints_violations(best_timetable, E, M, N, K)
    best_fitness = hard * 1000 + soft

    print("Best Timetable:", best_timetable)
    print(f"Fitness of Best Solution: {best_fitness}")
    print(f"  Hard Constraint Violations: {hard}")
    print(f"  Soft Constraint Violations: {soft}")
    print(f"Best Parameters: {best_params}")

    plt.figure(figsize=(8, 4))
    plt.plot(fitness_per_gen)
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.title('Fitness Over Generations')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()