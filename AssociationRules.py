# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
def recursion(input_li, level):
    if level == 0:
        return []

    output = []
    for i in range(len(input_li)-level+1):
        rec = recursion(input_li[i+1:], level-1)
        c = input_li[i]
        if rec == []:
            output.append([c])
            continue
        for r in rec:
            r.append(c)
            output.append(r)

    return output




def combinations(input_list, n):
    return recursion(input_list, n)



class association:
    """
    Holds X-> Y
    """
    def __init__(self, li, top, confi, supp):
        self.X = list(li)
        self.Y = top
        self.confi = confi
        self.support = supp

    def __str__(self):

        slen = len('%.*f' % (4, self.confi))
        confi = str(self.confi)[:slen]

        slen = len('%.*f' % (4, self.support))
        supp = str(self.support)[:slen]

        s = str(self.X) + '->' + str(self.Y) + ' Confi ' + confi + ' Supp ' + supp
        return s

def sortby_sup(input_li, start, end):
    for i in range(start, end):
        k = i;
        for j in range(i+1, end):
            if input_li[k].support < input_li[j].support:
                k = j
        temp = input_li[k]
        input_li[k] = input_li[i]
        input_li[i] = temp


def sort(input_li):
    for i in range(0, len(input_li)):
        k = i;
        for j in range(i+1, len(input_li)):
            if input_li[k].confi < input_li[j].confi:
                k = j
        temp = input_li[k]
        input_li[k] = input_li[i]
        input_li[i] = temp

    index = 0
    end = 0
    for i in range(len(input_li)):
        if input_li[i].confi == input_li[index].confi:
            end = end+1
        else:
            if index != end:
                sortby_sup(input_li, index, end)
            index = i
            end = i




def AssociationRules(data_matrix, words, topics_matrix, support):

    support = support * len(data_matrix)
    support = int(support)
# len(data_matrix) gives the percentage of transactions. Multiplied by len of data_matrix, which number of tramsaction, gives the absolute number.
    topics = []
    for t in topics_matrix:
        topics = list(set(topics) | set(t))

    data_refined = []

    for d in data_matrix:
        tem = []
        for i in range(len(d)):
            if(d[i] != 0):
                tem.append(i)
        data_refined.append(tem)


    in_list = []
    in_topics = list(topics)

    topics_with_support = []

    for top in in_topics:
        count  = 0
        for arti_top in topics_matrix:
            if top in arti_top:
                count = count + 1
        if count > support:
            topics_with_support.append(top)


    out_list = []
    word_list = []

    for word in words:
# List of all the words
        word_list.append(word)

    for i in range(len(word_list)):
#Create a list of lists. Where each list is the word index we are checking in the Article
        in_list.append([i])


    #dictionary, where key is the topic and value is word list only if they clear support.

    dictionary = {}
    support_dictionary = {}
    confidence_dictionary = {}
    return_list = []

    comb = 2;  # Number of combinations of words for next iteration.

    while(in_list != []):

#Maintain the total number of occurences of each list of words, we are interested in
        num_occurences = []

# Run through the list of word indexes, for each entry check how many articles contain those word indexes.
        for li in in_list:
            count = 0
            for data in data_refined:
                intersec = set(data) & set(li)
                intersec = list(intersec)
                if intersec == li:
                    count = count + 1
            num_occurences.append(count)

        in_list_with_support = []

        for i in range(len(num_occurences)):
            if num_occurences[i] > support:
                in_list_with_support.append(in_list[i])

        # Add only those words, which make the mark in this iteration
        next_iter_words = set()

        # Run through all the articles matching for every pair of
        # word -> topics. Create a two dimentional matrix [words][topics] holding the count
        # of articles containing both(word and topic).


        matrix = [[0 for x in range(len(topics_with_support))] for x in range(len(in_list_with_support))]

        for w in range(len(in_list_with_support)):
            word = in_list_with_support[w]
            for t in range(len(topics_with_support)):
                topicc = topics_with_support[t]
                count = 0;
                word_count = 0
                for a in range(len(data_refined)):
                    word_inster = list(set(data_refined[a]) & set(word))
                    topic_inter = list(set(topics_matrix[a]) & set([topicc]))
                    if word_inster == word and topic_inter == [topicc]:
                        count = count + 1

                    if word_inster == word:
                        word_count = word_count + 1


                matrix[w][t] = count
                if count > support:
                    for it in word:
                        next_iter_words.add(it)

                    if topicc not in dictionary:
                        dictionary[topicc] = []
                        support_dictionary[topicc] = []
                        confidence_dictionary[topicc] = []

                    dictionary[topicc].append(word)
                    support_dictionary[topicc].append((count*1.0)/len(data_refined))
                    confidence_dictionary[topicc].append((count*1.0)/word_count)
                    result = association(word, topicc, ((count*1.0)/word_count), ((count*1.0)/len(data_refined)))
                    return_list.append(result)

        topics_with_support = []

        for k in dictionary.keys():
            topics_with_support.append(k)


        next_iter_words = list(next_iter_words)

        next_iter_words = combinations(next_iter_words, comb);

        comb = comb + 1
        in_list = next_iter_words

    sort(return_list)
    for re in return_list:
        n = []
        for w in re.X:
            n.append(word_list[w])
        re.X = n
        print re
    return return_list;

















