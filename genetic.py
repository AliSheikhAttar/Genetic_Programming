import numpy as np
import random, math, sys
import matplotlib.pyplot as plt

random.seed(3)

def Generic_Crossover_ValidTree(tree,index):
    counter = 1
    finish = index
    while(counter!=0):
        if(tree[finish] in operands):
            counter+=1
        else:
            counter -=1
        finish -=1

    return tree[finish+1:index+1]

def Cuttree(tree, length):
    counter = 1
    i = 0
    finish = len(tree)-1
    while(counter!=0 and i<= length):
        if(tree[finish] in operands):
            counter+=1
        else:
            counter -=1
        finish -=1
    return finish+1 #the first index which the subtree before it is a tree and has lenght equal or more than the given parameter length


def Target_Function(x):
    return x**2 + (x+2)/3

def Valid_Function(tree):
    mystack = []
    errors = []
    result = 0
    if(len(tree)>2):
        for i in range(len(inputs)):
            for x in tree:
                if(x not in operands):
                    mystack.append(x)
                else:
                    tmp1 = mystack.pop()
                    tmp2 = mystack.pop()

                    if(tmp1=='x'):
                        tmp1 = inputs[i]
                    if(tmp2=='x'):
                        tmp2 = inputs[i]
                    
                    tmp1 = tmp1
                    tmp2 = tmp2

                    if(x == '+'):
                        mystack.append(tmp1+tmp2)

                    elif(x == '-'):
                        mystack.append(tmp1-tmp2)

                    elif(x == '^'):

                        if(tmp1!=0):
                            if(tmp1>100):
                                tmp1 = 100
                            elif(tmp1<-100):
                                tmp1 = -100
                            if(tmp2>100):
                                tmp2 = 100
                            elif(tmp2<-100):
                                tmp2 = -100
                            resultofpower = tmp1**tmp2
                            if(abs(resultofpower)>1000000):
                                if(str(resultofpower)[0]=="-"):
                                    resultofpower = -100000
                                else:
                                    resultofpower = 100000
                        else:
                            resultofpower = 0
                        if(str(resultofpower)[0] == "-"): # for "complex" error
                            mystack.append(resultofpower)
                        else:
                            mystack.append(resultofpower)



                    elif(x == '*'):  
                        mystack.append(round(tmp1*tmp2,3))

                    elif(x == '/'):
                        if(tmp2!=0):
                            mystack.append(tmp1/tmp2)
                        else:
                            mystack.append(tmp1*100)
                        continue

                
            result = mystack.pop()

            try: #for overflow catch
                error = (abs(result-Target_Function(inputs[i])))**2
                error = round(error,3)
            except:
                error = sys.maxsize//1000
            errors.append(error)

    else:
        result = tree[-1]
        if(result!="x"):
            for i in range(len(inputs)):
                try: #for overflow catch
                    error = round((abs(result-Target_Function(inputs[i])))**2,3)
                except:
                    error = sys.maxsize//1000

                errors.append(error)

        else:
            for i in range(len(inputs)):
                
                result = inputs[i]

                try: #for overflow catch
                    error = round((abs(result-Target_Function(inputs[i])))**2,3)
                except:
                    error = sys.maxsize//1000

                errors.append(error)

    return sum(errors)/len(errors)


def Generating_Random(n):
    trees = []
    for i in range(n):
        operators_num = np.random.randint(0,7)
        if(operators_num==0):
            tree = []
            tree.append(np.random.randint(1,10))
            trees.append([tree,Valid_Function(tree)])
        else:
            counter = 1
            length = operators_num*2 + 1 
            tree = []
            tree.append(np.random.randint(1,10))
            tree.append(np.random.randint(1,10))
            while(len(tree)!= length and ((length-len(tree))>counter)):
                to_select = np.random.randint(6)
                if(counter <=0 or to_select%3 !=2): #add number or variable / to_select %3 == 0, 1
                    if(to_select==0 or to_select==3): #add number
                        tree.append(np.random.randint(1,10))
                        counter += 1
                    elif(to_select==1 or to_select==4): #add varaible
                        tree.append('x')
                        counter += 1
                    else:
                        to_select = np.random.randint(2) #not allowed to add operator/counter<=0
                        if(to_select==2):
                            tree.append(np.random.randint(1,10)) #add number
                            counter += 1
                        elif(to_select==5):
                            tree.append('x') #add variable
                            counter += 1

                else:
                    tree.append(np.random.choice(operands)) #add operator / to_select %3 == 2
                    counter -= 1
            if not(length-len(tree)>counter):
                for i in range(counter):
                    tree.append(np.random.choice(operands))
            trees.append([tree,Valid_Function(tree)])
    return trees


def Generic_Select(n):
    if(len(Generations)>1):
        Last_Gen = Generations[-1]
        Last_Gen.sort(key=lambda x : x[1])
        Next_Gen = []
        for i in range(int(0.65*n)):
            if(Last_Gen[i] not in Next_Gen):
                Next_Gen.append(Last_Gen[i])

        Previous_Gen = Generations[-2]
        Previous_Gen.sort(key=lambda x : x[1])
        for i in range(int(0.35*n)):
            if(Previous_Gen[i] not in Next_Gen):
                Next_Gen.append(Previous_Gen[i])

        Generations.append(Next_Gen)
    else:
        Last_Gen = Generations[-1]
        Last_Gen.sort(key=lambda x : x[1])
        Next_Gen = []
        for i in range(int(0.8*n)):
            if(Last_Gen[i] not in Next_Gen):
                Next_Gen.append(Last_Gen[i])

        Generations.append(Next_Gen)


def Generic_mutation(percentage,mutate_percentage):
    indexes2mutate = [int(x) for x in np.random.uniform(0,len(Generations[-1]),int(percentage*len(Generations[-1])))] #which trees to mutate
    for index in indexes2mutate:
        tomutate = Generations[-1][index][0]
        del Generations[-1][index]
        indexes2mutate = [int(x) for x in np.random.uniform(0,len(tomutate),int(mutate_percentage*len(tomutate)))] #which nodes to mutate
        for mutateindex in indexes2mutate:
            if(tomutate[mutateindex] not in operands): #mutate ints and variables
                to_select = np.random.randint(2)
                if(to_select==0):
                    tomutate[mutateindex] = np.random.randint(1,10) #change to number
                else:
                    tomutate[mutateindex] = 'x'                     #change to variable
            else:                                     #mutate operators
                tomutate[mutateindex] = np.random.choice(operands)

        # if([tomutate,Valid_Function(tomutate)] not in Generations[-1]):
        Generations[-1].append([tomutate,Valid_Function(tomutate)])

def Generic_Crossover(percentage):
    indexes2crossover = [int(x) for x in np.random.uniform(0,len(Generations[-1]),int(percentage*len(Generations[-1])))]
    if not(len(indexes2crossover)%2==0):#must be even because each time select two trees
        del indexes2crossover[np.random.randint(len(indexes2crossover))]
    while(len(indexes2crossover)%2==0 and len(indexes2crossover)!=0): #select two trees and remove them from choices
        index1 = np.random.choice(indexes2crossover)
        indexes2crossover.remove(index1)
        index2 = np.random.choice(indexes2crossover)
        indexes2crossover.remove(index2)

        to_select = np.random.randint(2)

        if(to_select==0): #add random subtree from first tree to a random leaf from second tree and add the produced tree to the last generation
            if(len(Generations[-1][index1][0])!=1): #if the first tree is a constant value, it doesn't have a operand
                index2pick1 = np.random.randint(len(Generations[-1][index1][0]))
                while(Generations[-1][index1][0][index2pick1] not in operands):
                    index2pick1 = np.random.randint(len(Generations[-1][index1][0]))
                offspring1 = Generic_Crossover_ValidTree(Generations[-1][index1][0],index2pick1)
                index2pick2 = np.random.randint(len(Generations[-1][index2][0]))
                while(Generations[-1][index2][0][index2pick2] in operands):
                    index2pick2 = np.random.randint(len(Generations[-1][index2][0]))
                offspring2 = Generations[-1][index2][0][0:index2pick2] + offspring1 + Generations[-1][index2][0][index2pick2+1::]
                if(len(offspring2)>15): #cut the tree if has a lenght more than 15
                    subtreeindex = Cuttree(offspring2,15) #return the first index of the tree wich satisfies the condition
                    if(subtreeindex!=0): #its not the same tree
                        Generations[-1].append([offspring2[subtreeindex::],Valid_Function(offspring2[subtreeindex::])])
                else:
                    Generations[-1].append([offspring2,Valid_Function(offspring2)])
            else: #the first tree is a single constant value so it doesn't have operand so cant any subtree from it to add to the second tree, so the crossover operation wont be performed
                continue
        else: #crossover two random nodes
            index2pick1 = np.random.randint(len(Generations[-1][index1][0])) #select random node from first tree
            if(Generations[-1][index1][0][index2pick1] in operands): #it's operand, swap with random operand from second tree
                if(len(Generations[-1][index2][0])!=1): #if the second tree is constant value, doesn't contain operand
                    index2pick2 = np.random.randint(len(Generations[-1][index2][0])) #until find node from second tree
                    while(Generations[-1][index2][0][index2pick2] not in operands):
                        index2pick2 = np.random.randint(len(Generations[-1][index2][0]))
                    tmp1 = Generations[-1][index1][0][index2pick1]
                    Generations[-1][index1][0][index2pick1] = Generations[-1][index2][0][index2pick2]
                    Generations[-1][index2][0][index2pick2] = tmp1
                else:#the second tree is a constant that doesn't contain operand so crossover won't be operated
                    continue
            else:                                                   #it's not operand, swap with random number or variable from second tree
                index2pick2 = np.random.randint(len(Generations[-1][index2][0])) #select random node from  second tree
                while(Generations[-1][index2][0][index2pick2] in operands): #until find number or variable from second tree
                    index2pick2 = np.random.randint(len(Generations[-1][index2][0]))
                tmp1 = Generations[-1][index1][0][index2pick1]
                Generations[-1][index1][0][index2pick1] = Generations[-1][index2][0][index2pick2]
                Generations[-1][index2][0][index2pick2] = tmp1


def Pick_the_Best(n):
    Generations[-1].sort(key=lambda x: x[1])
    the_best = Generations[-1][0]
    for i in range(n):
        Generations[i].sort(key=lambda x: x[1])
        if(Generations[i][0][1]<= the_best[1]):
            the_best = Generations[i][0]

    return the_best





input_size = 70
Generations = []
inputs = np.random.uniform(-0.01,0.01,input_size)
inputs = [round(x,3) for x in inputs]
operands = ['+' , '^','-', '*', '/']
Generations_Num = 100
first_population = 50
in_howmany_generations = 3


Generations.append(Generating_Random(first_population))
for i in range(Generations_Num-1): #generations 50 to up results to overflow in error calculations segment,generation 74 to up result to constant values in generations
    Generic_Select(int(0.8*(first_population-(i//10))))
    Generic_mutation(0.7, 0.6)
    Generic_Crossover(0.8)


theBestTree = Pick_the_Best(in_howmany_generations)

print(f"the mean squared error is : {theBestTree[1]}")

the_bests_errors = []

for gen in Generations:
    gen.sort(key= lambda x: x[1])
    the_bests_errors.append(gen[0][1])

generation_number = [i for i in range(Generations_Num)]

plt.plot(generation_number, the_bests_errors)
plt.show()