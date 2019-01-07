# -*- coding: utf-8 -*-

import random

import numpy as np
import csv

from deap import base
from deap import creator
from deap import tools

f=open("Defence.csv","r")
jlines=[[elm for elm in v] for v in csv.reader(f)]
score_def=[]
myscore_def=[]
for i in range(1,len(jlines)):
    score_def.append(float(jlines[i][2])+float(jlines[i][3]))
    myscore_def.append(float(jlines[i][4])*float(jlines[i][5]))
f.close()
    
f=open("Attack.csv","r")
jlines=[[elm for elm in v] for v in csv.reader(f)]
score_att=[]
myscore_att=[]
for i in range(1,len(jlines)):
    score_att.append(float(jlines[i][2])+float(jlines[i][3]))
    myscore_att.append(float(jlines[i][4])/2*float(jlines[i][5]))
f.close()

ATTACK=True

if ATTACK:
    n_gene=len(score_att)
else:
    n_gene=len(score_def)

def create_ind_uniform():
    ind = []
    for i in range(n_gene):
        ind.append(0)
    num=np.random.permutation(n_gene)
    for i in range(5):
        if i==0:
            ind[num[i]]=2
        else:
            ind[num[i]]=1
    return ind
    
def cxTwoPointCopy(ind1, ind2):
    size = len(ind1)
    while(True):
        cxpoint1 = random.randint(1, size-1)
        cxpoint2 = random.randint(1, size-1)
        if (ind1[cxpoint1]==2 and ind2[cxpoint1]==0) and (ind2[cxpoint2]==2 and ind1[cxpoint2]==0):
            break
        if (ind1[cxpoint1]==2 and ind2[cxpoint1]==1) and (ind2[cxpoint2]==2 and ind1[cxpoint2]==1):
            break
        if (ind1[cxpoint1]==1 and ind2[cxpoint1]==0) and (ind2[cxpoint2]==1 and ind1[cxpoint2]==0):
            break
        if (ind1[cxpoint1]==1 and ind2[cxpoint1]==1) and (ind2[cxpoint2]==1 and ind1[cxpoint2]==1):
            break
        if (ind1[cxpoint1]==1 and ind2[cxpoint1]==1) and (ind2[cxpoint2]==0 and ind1[cxpoint2]==0):
            break
        if (ind1[cxpoint1]==0 and ind2[cxpoint1]==0) and (ind2[cxpoint2]==1 and ind1[cxpoint2]==1):
            break
    
    a1=ind1[cxpoint1]
    a2=ind2[cxpoint1]
    b1=ind1[cxpoint2]
    b2=ind2[cxpoint2]
    
    ind1[cxpoint1]=a2
    ind2[cxpoint1]=a1
    ind2[cxpoint2]=b1
    ind1[cxpoint2]=b2
    
    #ind1[cxpoint1], ind2[cxpoint2] = ind2[cxpoint1], ind1[cxpoint2]
    
    return ind1, ind2

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_bool", create_ind_uniform)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_bool)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evalOneMax(individual):
    if ATTACK:
        score=np.array(score_att)
        myscore=np.array(myscore_att)
    else:
        score=np.array(score_def)
        myscore=np.array(myscore_def)
    b=np.array(individual)
    for i in range(n_gene):
        if b[i]==2:
            twice=i
            break
    fx=b*score
    fx[twice]=fx[twice]/2
    fx=np.sum(fx)
    
    gx=b*score*myscore
    #gx[twice]+=b[twice]*myscore[twice]
    gx=np.sum(gx)
    
    #gx=(b[twice]*score[twice]*myscore[twice])
    
    return gx,

toolbox.register("evaluate", evalOneMax)
toolbox.register("mate", cxTwoPointCopy)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    random.seed()
    pop = toolbox.population(n=300)
    CXPB, MUTPB, NGEN = 0.5, 0.2, 100
    
    print("Start of evolution")
    fitnesses = list(map(toolbox.evaluate, pop))
    
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    print("  Evaluated %i individuals" % len(pop))
    
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        
        pop[:] = offspring
        
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    if ATTACK:
        print(score_att)
    else:
        print(score_def)
if __name__ == "__main__":
    
    a=3
    b=0
    c=4
    d=0
    if ((1<=a and a<=5) and (b==0)) and ((1<=c and c<=5) and (d==0)):
        print("aaa")
    
    #main()