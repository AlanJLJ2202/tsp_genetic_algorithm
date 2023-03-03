element_lenght, start_point, end_point = map(int, input().split())
elements = []

for i in range(2):
    element = input().split(' ')
    for j in range(len(element)):
        element[j] = int(element[j])
    
    elements.append(element)

#print("-----------------------------------")

def PMX(solution1, solution2, cut1, cut2, size):
    child = []

    for i in range(size):
        child.append(None)
    
    if cut1 > cut2:
        cut1,cut2 = cut2,cut1
    

    child[cut1:cut2+1] = solution1[cut1:cut2+1]
    print("child" , child)
    parent2 = solution2[cut1:cut2+1]
    print("parent2", parent2)

    for i in range(len(child)):
        for j in range(len(parent2)):
            if parent2[j] not in child:
                idxchild = solution2.index(solution1[solution2.index(parent2[j])])
                print(child)
                while child[idxchild] != None:
                    idxchild = solution2.index(solution1[idxchild])
                child[idxchild] = parent2[j]
        
        if child[i] == None:
            child[i] = solution2[i]
            
    return child

child = PMX(elements[0], elements[1], start_point, end_point, element_lenght)
text = " ".join(str(val) for val in child) 
print(text)