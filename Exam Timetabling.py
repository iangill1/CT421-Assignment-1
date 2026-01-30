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

def run_ga(file_path): 
    POP_SIZE = 1000
    GENERATIONS = 500
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.1

    N, K, M, E = read_instance(file_path)
    population = [[random.randint(0, K - 1) for _ in range(N)] for _ in range(POP_SIZE)]
    best_fitness_per_gen = []

    for generation in range(GENERATIONS):
        fitnesses = [evaluate_fitness(t, E, M, N, K) for t in population]
        best_fitness_per_gen.append(min(fitnesses))

        new_population = []

        while len(new_population) < POP_SIZE:
            parent1 = tournament_selection(population, fitnesses)
            parent2 = tournament_selection(population, fitnesses)
            child1, child2 = crossover(parent1, parent2, CROSSOVER_RATE)
            mutate(child1, K, MUTATION_RATE)
            mutate(child2, K, MUTATION_RATE)
            new_population.append(child1)
            if len(new_population) < POP_SIZE:
                new_population.append(child2)

        population = new_population

    fitnesses = [evaluate_fitness(t, E, M, N, K) for t in population]
    best_timetable = population[fitnesses.index(min(fitnesses))]
    return best_timetable, best_fitness_per_gen, E, M, N, K

def main(): 
    best_timetable, fitness_per_gen, E, M, N, K = run_ga('tinyexample.txt')

    hard = hard_constraints_violations(best_timetable, E, M, N)
    soft = soft_constraints_violations(best_timetable, E, M, N, K)
    best_fitness = hard * 1000 + soft

    print("Best Timetable:", best_timetable)
    print(f"Fitness of Best Solution: {best_fitness}")
    print(f"  Hard Constraint Violations: {hard}")
    print(f"  Soft Constraint Violations: {soft}")

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