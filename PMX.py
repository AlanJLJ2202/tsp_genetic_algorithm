
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
        child.append(0)
    
    if cut1 > cut2:
        aux = 0
        cut1 = aux
        cut1 = cut2
        cut2 = aux

    child[cut1:cut2+1] = solution1[cut1:cut2+1]

    #print('Child =', child)
    aux = solution2[cut1:cut2+1]
    #print('Aux =', aux)

    for i in range(len(child)):
        for j in range(len(aux)):
            if aux[j] not in child:
                idxchild = solution2.index(solution1[solution2.index(aux[j])])
                if child[idxchild] == 0:
                    child[idxchild] = aux[j]
                else:
                    child[solution2.index(solution1[idxchild])] = aux[j]
        if child[i] == 0:
            child[i] = solution2[i]
    
    return child

child = PMX(elements[0], elements[1], start_point, end_point, element_lenght)
text = " ".join(str(val) for val in child) 
print(text)