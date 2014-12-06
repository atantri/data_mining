import math
from scipy import stats





def variance(lis):
    var = 0.0;
    mean = 0.0;
    for i in lis:
        mean = mean + i;

    mean = (mean)/(len(lis))
    for i in lis:
        diff = mean - i;
        var = var + diff * diff

    return(var/len(lis))



def entropy(results, list_articles, num_clusters):
    clusters = []
    for i in range(num_clusters):
        clusters.append([])

    
    index = 0;
    for i in results:
        clusters[i].append(index)
        index = index + 1

    ent = []
    cluster_size = []
    for cluster in clusters:
        print "Size of cluster="+str(len(cluster))
        
        all_topics = set()
        topics_count = {}
        for i in cluster:
            article = list_articles[i]
            
            for topic in article.topics:
                all_topics.add(topic)
                
        for topics in all_topics:
            topics_count[topics] = 0;
        print ("Number of topics in cluster="+str(len(all_topics)))
        for i in cluster:
            article = list_articles[i]
            for topic in article.topics:
                for s in all_topics:
                    if s in topic:
                        topics_count[s] = topics_count[s] + 1
        num_articles = len(cluster)
        cluster_size.append(num_articles)
        cluster_entropy = 0.0;
        for s in all_topics:
            p = float(topics_count[s])/num_articles
            if p > 0 and p <= 1:
                p = -p * math.log(p,2)
                cluster_entropy = cluster_entropy + p

        ent.append(cluster_entropy)
    print("Number of elements in each cluster")
    print cluster_size
    print("Entropy for each cluster");
    print ent
    print("Variance")
    print (variance(cluster_size))





