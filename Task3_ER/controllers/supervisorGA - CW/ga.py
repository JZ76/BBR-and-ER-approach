import numpy
import random


def population_reproduce(genotypes,elite):
    ### Define crossover rate (integer number between 0 and 100)
    cp = 50
    genotypes_not_ranked = genotypes
    # Rank: lowest to highest fitness
    genotypes = rankPopulation(genotypes)
    # Initiate loop to create new population
    population_size = len(genotypes)
    new_population = []
    # Backwards: from highest to lowest fitness
    for individual in range(population_size,0,-1):
        # Clone the elite individuals
        if population_size-individual < elite:
            new_population.append(genotypes[individual-1][0])
        elif random.randint(1, 100) > cp:
            new_population.append(genotypes[individual-1][0])
        else:  
            # Generate the rest of the population by using the genetic operations 
            parent1 = selectParent(genotypes_not_ranked)
            parent2 = selectParent(genotypes_not_ranked)
            # Apply crossover
            child = crossover(parent1,parent2)
            # Apply mutation
            offspring = mutation(child)
            new_population.append(numpy.array(offspring))
                  
    return new_population

def rankPopulation(genotypes):
    # Rank the populations using the fitness values (Lowest to Highest)
    genotypes.sort(key=lambda item: item[1])  
    return genotypes
    
def getBestGenotype(genotypes):
    return rankPopulation(genotypes)[-1]
    
def getAverageGenotype(genotypes):
    sum = 0.0
    for g in range(0,len(genotypes)-1):
        sum = sum + genotypes[g][1]
    return sum / len(genotypes)
    
def selectParent(genotypes):
    # Tournament Selection  
    # Select a few individuals of the population randomly
    group = []
    population_size = len(genotypes)
    number_individuals = 5
    for selected in range(0,number_individuals-1):
        group.append(genotypes[random.choice([0, population_size-1])])
    # Then, select the best individual of this group
    group_ranked = rankPopulation(group)     
    return group_ranked[-1]
    
def crossover(parent1,parent2):
    child = []
    # Center
    crossover_point = int(len(parent1[0])/2)
    for gene in range(len(parent1[0])):
        # The new offspring will have its first half of its genes taken from one parent
        if gene < crossover_point:
            child.append(parent1[0][gene])
        else:
            child.append(parent2[0][gene])       

    return child      
    
def mutation(child):
    # Changes a single gene randomly
    after_mutation = []
    ### Define mutation percentage (integer number between 0 and 100)
    mp = 25
    for gene in range(len(child)):
        if random.randint(1, 100) < mp:
            # The random value to be added to the gene
            random_value = numpy.random.uniform(-1.0, 1.0, 1)
            temp = child[gene] + random_value[0]
            # Clip
            if temp < -1: temp = -1
            elif temp > 1: temp = 1
            after_mutation.append(temp)
        else:
            after_mutation.append(child[gene])         
    return after_mutation