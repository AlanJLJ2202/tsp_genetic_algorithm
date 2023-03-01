
element_lenght, start_point, end_point = map(int, input().split())
elements = []


for i in range(2):
    element = input().split(' ')
    for j in range(len(element)):
        element[j] = int(element[j])
    
    elements.append(element)

print(elements)


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

    print('Child =', child)
    aux = solution2[cut1:cut2+1]
    print('Aux =', aux)

    for i in range(len(child)):
        for j in range(len(aux)):
            if aux[j] not in child:
                #print(aux[j])
                idx = solution2.index(aux[j])
                print('IDX', idx)
                idxchild = solution2.index(solution1[idx])
                print('IDXCHILD', idxchild)
                child[idxchild] = aux[j]
                
    print('NEW CHILD', child)
            
    if solution2[i] not in child:
        print(solution2[i])
        
        ''''
        if solution2[i] not in child:
            child[i] = solution2[i]
        else:
            child[i] = solution1[i]
        '''

    #print(child)

    return child

PMX(elements[0], elements[1], start_point, end_point, element_lenght)
