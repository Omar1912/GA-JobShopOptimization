#Omar Husain 1212739
#Malak Moqbel 1210608
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Function to get integer input from the user with optional maximum value validation
def get_int_input(prompt, max_val=None):
    while True:
        try:
            value = int(input(prompt))
            if value > 0 and (max_val is None or value <= max_val):
                return value
            elif max_val is not None and value > max_val:
                print(f"Please enter a value less than or equal to {max_val}.")
            else:
                print("Please enter a positive integer greater than 0.")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

# Function to gather user input for jobs, machines, and operations
def get_user_input():
    operations = {}
    num_jobs = get_int_input("Enter the number of jobs: ")
    num_machines = get_int_input("Enter the number of available machines: ")
    total_operations = 0
    for job in range(1, num_jobs + 1):
        job_operations = []
        num_operations = get_int_input(f"Enter the number of operations for Job {job}: ")
        total_operations += num_operations
        for op in range(1, num_operations + 1):
            machine = get_int_input(f"Enter the machine number for Job {job}, Operation {op} (1-{num_machines}): ", max_val=num_machines)
            time = get_int_input(f"Enter time required for Job {job}, Operation {op}: ")
            job_operations.append((f'M{machine}', f'Job{job}', f'Operation{op}', time))
        operations[f'Job{job}'] = job_operations
    return operations, num_machines, total_operations

# Function to create a random chromosome (a possible solution)
def create_random_chromosome(operations):
    job_keys = list(operations.keys())
    random.shuffle(job_keys)  # Shuffle job keys to randomize the order of jobs
    chromosome = []
    scheduled_operations = [(job, 0) for job in job_keys]  # Initialize each job with its first operation
    while scheduled_operations:
        for job_index in range(len(scheduled_operations)):
            job, op_index = scheduled_operations[job_index]
            if op_index < len(operations[job]):
                chromosome.append(operations[job][op_index])  # Append the operation to the chromosome
                scheduled_operations[job_index] = (job, op_index + 1)  # Move to the next operation of the job
        scheduled_operations = [(job, op_index) for job, op_index in scheduled_operations if op_index < len(operations[job])]
    return chromosome



            


# Function to generate the initial population of chromosomes
def generate_initial_population(size, operations):
    population = []
    for _ in range(size):
        chromosome = create_random_chromosome(operations)
        population.append(chromosome)
    return population
# Function to calculate the makespan (total time) of a chromosome
def calculate_makespan(chromosome):
    machine_end_times = {}  # Track the end time for each machine
    job_end_times = {}  # Track the end time for each job
    for machine, job, operation, time in chromosome:
        start_time = max(machine_end_times.get(machine, 0), job_end_times.get(job, 0))  # Get the earliest possible start time
        end_time = start_time + time  # Calculate end time for the operation
        machine_end_times[machine] = end_time  # Update machine end time
        job_end_times[job] = end_time  # Update job end time
    return max(machine_end_times.values())  # Makespan is the latest end time among all machines

# Function to calculate the fitness of a chromosome (inverse of makespan)
def calculate_fitness(chromosome):
    return 1 / calculate_makespan(chromosome)

# Function to select the two best chromosomes from the population
def select_two_best(population):
    sorted_population = sorted(population, key=lambda x: calculate_fitness(x), reverse=True)
    return sorted_population[:2]  # Return the two chromosomes with the highest fitness

# Function to perform order crossover between two parent chromosomes
def order_crossover(parent1, parent2):
    size = len(parent1)
    point1, point2 = sorted(random.sample(range(size), 2))  # Select two crossover points
    child1_part = parent1[point1:point2]  # Get the part of parent1 between the crossover points
    child2_part = parent2[point1:point2]  # Get the part of parent2 between the crossover points

    def fill_child(part, parent):
        child = part[:]
        parent_index = point2
        while len(child) < size:
            operation = parent[parent_index % size]  
            if operation not in child:  # Ensure no duplicates in child
                child.append(operation)
            parent_index += 1
        return child

    child1 = fill_child(child1_part, parent2)
    child2 = fill_child(child2_part, parent1)
    return child1, child2

# Function to check if a chromosome is valid (operations are in the correct order)
def is_valid_chromosome(chromosome, operations):
    last_indices = {}
    for operation in chromosome:
        machine, job, op, time = operation
        op_index = operations[job].index(operation)
        if job in last_indices:
            if op_index <= last_indices[job]:  # Check if the operation is in the correct order
                return False
        last_indices[job] = op_index  # Update the last operation index for the job
    return True

# Function to mutate a chromosome
def mutate(chromosome, operations, mutation_rate):
    if random.random() > mutation_rate:  # Only mutate with a certain probability
        return chromosome, chromosome

    machine_ops = {}
    for op in chromosome:
        machine = op[0]
        if machine not in machine_ops:
            machine_ops[machine] = []
        machine_ops[machine].append(op)

    eligible_machines = [m for m in machine_ops if len(machine_ops[m]) > 1]  # Find machines with more than one operation
    if not eligible_machines:
        return chromosome, chromosome

    machine = random.choice(eligible_machines)  # Select a random eligible machine
    op1, op2 = random.sample(machine_ops[machine], 2)  # Select two random operations from the machine

    new_chromosome = []
    for op in chromosome:
        if op == op1:
            new_chromosome.append(op2)  # Swap operations
        elif op == op2:
            new_chromosome.append(op1)
        else:
            new_chromosome.append(op)

    if is_valid_chromosome(new_chromosome, operations):  # Ensure the mutated chromosome is valid
        return chromosome, new_chromosome
    return chromosome, chromosome

# Function for the genetic algorithm loop
def genetic_algorithm_loop(initial_population, operations, num_crossovers, mutation_rate):
    population = initial_population
    for _ in range(num_crossovers):
        best_two = select_two_best(population)  # Select the two best chromosomes
        new_chromosome1, new_chromosome2 = order_crossover(best_two[0], best_two[1])  # Perform crossover
        before_mutation1, new_chromosome1 = mutate(new_chromosome1, operations, mutation_rate)  # Mutate the first child
        before_mutation2, new_chromosome2 = mutate(new_chromosome2, operations, mutation_rate)  # Mutate the second child

        if is_valid_chromosome(new_chromosome1, operations):  # Check if the first child is valid
            population.append(new_chromosome1)

        if is_valid_chromosome(new_chromosome2, operations):  # Check if the second child is valid
            population.append(new_chromosome2)

        population = select_two_best(population)  # Keep only the two best chromosomes in the population

    best_chromosome = select_two_best(population)[0]  # Return the best chromosome after all iterations
    return best_chromosome

# Function to calculate the schedule from a chromosome
def calculate_schedule(chromosome):
    machine_end_times = {}
    job_end_times = {}
    schedule = []
    for machine, job, operation, time in chromosome:
        start_time = max(machine_end_times.get(machine, 0), job_end_times.get(job, 0))  # Get the earliest possible start time
        end_time = start_time + time  # Calculate end time for the operation
        schedule.append((machine, job, operation, start_time, end_time))  # Append the schedule entry
        machine_end_times[machine] = end_time  # Update machine end time
        job_end_times[job] = end_time  # Update job end time
    return schedule

# Function to plot a Gantt chart from the schedule
def plot_gantt_chart(schedule, best_chromosome):
    fig, ax = plt.subplots()
    job_colors = {'Job1': 'blue', 'Job2': 'lightblue', 'Job3': 'orange', 'Job4': 'red','Job5': 'pink'}  # Define colors for each job

    for machine, job, operation, start, end in schedule:
        ax.add_patch(mpatches.Rectangle((start, int(machine[1:])), end - start, 0.8, facecolor=job_colors[job]))
        ax.text(start + (end - start) / 2, int(machine[1:]) + 0.4, f'{job}-{operation}', ha='center', va='center', color='white')

    ax.set_ylim(0.5, max(int(machine[1:]) for machine, _, _, _, _ in schedule) + 1.5)
    ax.set_xlim(0, max(end for _, _, _, _, end in schedule))
    ax.set_xlabel('Time')
    ax.set_ylabel('Machine')
    ax.set_yticks([i for i in range(1, max(int(machine[1:]) for machine, _, _, _, _ in schedule) + 1)])
    ax.set_yticklabels([f'M{i}' for i in range(1, max(int(machine[1:]) for machine, _, _, _, _ in schedule) + 1)])
    ax.grid(True)
    ax.set_facecolor('#22334D')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    plt.title(f'Best Chromosome', color='white', fontsize=14, weight='bold')
    fig.patch.set_facecolor('#22334D')
    plt.show()

# Main function to run the genetic algorithm and plot the results
def main():
    operations, num_machines, total_operations = get_user_input()
    population_size = total_operations * 3  # Set population size equal to the total number of operations for all jobs * 3
    mutation_rate = 0.001  # Set mutation rate directly
    
    initial_population = generate_initial_population(population_size, operations)
    best_chromosome = genetic_algorithm_loop(initial_population, operations, 30, mutation_rate)
    best_fitness = calculate_fitness(best_chromosome)
    makespan = calculate_makespan(best_chromosome)
    schedule = calculate_schedule(best_chromosome)

    print("\nChromosome with the best fitness:")
    print(f"Chromosome: {best_chromosome}")
    print(f"Fitness: {best_fitness}")
    print(f"Makespan: {makespan}")

    print("\nGantt Chart Schedule:")
    for entry in schedule:
        print(entry)

    plot_gantt_chart(schedule, best_chromosome)

if __name__ == "__main__":
    main()
