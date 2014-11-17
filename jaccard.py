
def jaccardSimilarity(data_matrix):
    print "Entered Jaccard"
    sets = []

    for i in range(len(data_matrix)):
        sets.append([])
        for j in range(len(data_matrix[i])):
            if data_matrix[i][j] != 0:
                sets[i].append(j)


    jaccard_matrix = [];

    for i in range(len(data_matrix)):
        jaccard_matrix.append([])


    for i in range(len(jaccard_matrix)):
        jaccard_matrix[i].append(1)
        for j in range(i+1, len(jaccard_matrix)):
            union = set(sets[i]) | set(sets[j])
            intersection = set(sets[i]) & set(sets[j])
            value = (len(intersection) * 1.0) / len(union)
            jaccard_matrix[i].append(value)
            jaccard_matrix[j].append(value)
    print "Done"

    return jaccard_matrix




