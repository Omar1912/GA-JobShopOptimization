# GA-JobShopOptimization

This repository contains the source code for Optimizing Job Shop Scheduling in a Manufacturing Plant using a Genetic Algorithm (GA). This project was developed for the ENCS3340 Artificial Intelligence course.

## Project Overview
The project addresses the job shop scheduling problem, where a set of jobs must be assigned to machines in an optimal sequence to minimize total production time (makespan) and improve throughput. The challenge lies in determining the order of tasks while considering machine availability and dependencies between operations.

## Solution Approach
A `Genetic Algorithm` (GA) is employed to find near-optimal schedules. GAs are based on the principles of evolution, making them suitable for complex optimization tasks. Key components include:

- **Chromosome Representation** : A solution (schedule) is encoded as a chromosome.
- **Crossover and Mutation**: Methods to combine and vary solutions to explore the search space.
- **Fitness Function** : Evaluates each schedule based on its total production time.
## Data Structures & Algorithms
- **Lists** : Used to store jobs, machines, and their sequences.
- **Dictionaries**: Maps jobs to their sequences of operations.
- **Genetic Algorithm**: Implements selection, crossover, mutation, and fitness evaluation to evolve better schedules.
## Key Functions
- `generate_population()`: Creates an initial population of random schedules.
- `fitness_function()`: Evaluates how good a schedule is based on makespan.
- `crossover()`: Combines two schedules to produce new offspring.
- `mutation()`: Introduces random changes to a schedule for variation.
- `run_genetic_algorithm()`: Main function that iterates over generations to evolve the best schedule.





