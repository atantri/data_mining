import urllib
import nltk
import math
import Orange
import sklearn
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from HTMLParser import HTMLParser
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import operator
import collections
from collections import OrderedDict
from sklearn import tree
import timeit
from timeit import default_timer
import sklearn.cluster as clust
import random
from entropy import entropy


#Authors:Aneesh Tantri,Chandhan DS
class global_word_attributes:
	"""
	Count here holds the number of articles with
	the presence. it is part of global dictionary
	"""
	def __init__(self, count,wc, idf, tf_idf):
		self.art_count = count
		self.idf = idf
		self.tf=0
		self.tf_idf = tf_idf
		self.wrd_count=wc#added this to store global count

	def __str__(self):
		return str(self.art_count)

class local_word_attributes:
	"""
	Count here holds the number of times the word occurs
	in the article. Local dictionary for the given article.
	"""
	def __init__(self, count):
		self.wrd_count = count
		self.term_fre = 0


	def __str__(self):
		return str(self.wrd_count)

class corpus:
	def __init__(self):
		self.list_articles = []
		self.raw_dictionary ={}
		self.sortedDictionary=OrderedDict()
		self.tfIdfDict=OrderedDict()
		self.globalWordCount=0
		self.topicsList=[]
		self.placesList=[]
		self.articleTest=[]
		self.articleTrain=[]
		self.topicsMap={}
		self.articleMap={}
		self.precision=0
		self.recall=0
		self.fMeasure=0
		self.topicsTestMap={} #testing from articles with topics
		self.articleTrainTest=[]
		self.pre=0
	def is_stop_word(self, word):
		"""
		Returns true if the word is a stop word
		"""
		stop = stopwords.words('english')
		try:
			i = stop.index(word.lower())
		except ValueError:
			i = -1

		if(i == -1):
			return False
		else:
			return True
	def calcCS(self,resultTopicsMap,method):
		fp=0
		fn=0
		tp=0
		tn=0
		bfp=0
		bfn=0
		btp=0
		btn=0
		n=len(self.articleTrainTest)

		artList=self.topicsTestMap["earn"]
		for art in self.articleTrainTest:
			if art in artList:
				btp+=1;
			else:
				bfp+=1

		for topic in self.topicsTestMap:
			artList=self.topicsTestMap[topic]
			tempPred=0

			resArtList=[]

			if topic !="earn":
				bfn+=len(artList)
				btn+=n-len(artList)

			if(topic in resultTopicsMap):

				resArtList=resultTopicsMap[topic]
			for art in artList:
				if art in resArtList:
					tempPred+=1
					tp+=1;
				else:
					tempPred+=1
					fn+=1;


			for art in resArtList:
				if art not in artList:
					tempPred+=1
					fp+=1;
			tn+=n-tempPred


		self.precision=float(tp)/(tp+fp)
		self.recall=float(tp)/(tp+fn)
		self.fMeasure=2*self.precision*self.recall/(self.precision+self.recall)
		self.accuracy=float(tp+0.8*tn)/(tp+tn+fp+fn)
		bPrec=float(btp)/(btp+bfp)
		bRec=float(btp)/(btp+bfn)
		bFmeasure=2*bPrec*bRec/(bRec+bPrec)
		bAcc=(btp+0.8*btn)/(btp+btn*bfp+bfn)
		print method+": Accuracy="+str(self.accuracy)+" Precision= "+str(self.precision)+" Recall= "+str(self.recall)+" F Measure= "+str(self.fMeasure)
		print "\nBaseline guess for topic as earn: Accuracy:"+str(bAcc)+" Precision= "+str(bPrec)+" Recall= "+str(bRec)+" F Measure= "+str(bFmeasure)+"\n\n"
	def genTopicTestMap(self,X,Y,X_test):
		n=0
		size=0.8*len(self.topicsMap)
		for art in self.articleTrain:
			if(n<size):
				X.append(art.tfidfFeatureVector)
				#X.append(art.featureVector)

				yEl=[];
				i=0
				for t in self.topicsList:

					if(t in art.topics):
						yEl.append(i)
					i+=1

				Y.append(yEl)
			else:
				for t in art.topics:
					if t not in self.topicsTestMap:
						self.topicsTestMap[t]=[]
					self.topicsTestMap[t].append(art)
				X_test.append(art.tfidfFeatureVector)
				self.articleTrainTest.append(art)
			n+=1

	def classify(self,classifier,method,X,Y,X_test):



		start=default_timer()
		classifier.fit(np.array(X),Y)
		timeClassify=default_timer()-start+self.pre
		start=default_timer()
		Y_pred=classifier.predict(np.array(X_test))

		f=open('result'+method,'w')
		i=0
		correct=0
		resultTopicsMap={}
		for yRes in Y_pred:
			f.write(self.articleTrainTest[i].Id+"\t")
			for t in self.articleTrainTest[i].topics:
				f.write(t+"\t")
			f.write("vs\t")

			j=0
			if(len(yRes)==0):
				print "Wrong"

			for j in range(len(yRes)):
				predTop=self.topicsList[yRes[j]]
				if predTop in self.articleTrainTest[i].topics:
					correct+=1

				if(predTop not in resultTopicsMap):
					resultTopicsMap[predTop]=[]
				resultTopicsMap[predTop].append(self.articleTrainTest[i])
				f.write(predTop+"\t")
				j+=1

			"""
			for t in self.topicsList:
				b=0;
				if(yRes[j]==1):
					b=1
					f.write(t+"\t")
				if(b==0):
					print "missing"
				j+=1
			"""
			i+=1
			f.write("\n")
		#print "Acc="+str(float(correct)/len(self.articleTrainTest))

		self.calcCS(resultTopicsMap,method)
		timePred=default_timer()-start
		print method+" Offline cost="+str(timeClassify)+" Online Cost= "+str(timePred)
	
	def supportItem(self,itemList):
		count=0;
		
		for art in self.articleTrain:
			isIn=true;
			for word in itemList:
				if word not in art.word:
					isIn=false;
					break;
			if(isIn):
				count++;
		
		return count;
	self.itemIn=[];
	self.itemOut=[];
	self.globCount=0;
	def initApriori(self,s):
		for w in global_dic:
			self.globCount+=global_dic[w].wrd_count;
		for w in global_dic:
			if (1.0*global_dic[wrd_count])/globCount)>s:
				self.itemOut.append(w);
	
	def genAprioriPerm(self):
		self.itemIn=[];
		for item in self.itemOut:
			
	

	
	def aprioriGen(self,minSupport):
		
		for w in self.itemIn:
			if supportItem(w)>minSupport:
				itemOut.append(w);
				
		
		
		
	def aprioriTrim(self)
					
			
		
	def get_raw_data(self, parser):
		self.list_articles = parser.articleList;
		self.topicsList=parser.topicsList;
		self.placesList=parser.placesList
		self.articleMap=parser.articleMap

	def parse_raw_data(self, new_art):
		self.startClass=default_timer()
		tokenizer = RegexpTokenizer(r'\w+')
		tokens = tokenizer.tokenize(new_art.body)
		stemmer = LancasterStemmer()
		article_dic = new_art.words
		global_dic = self.raw_dictionary

		for word in tokens:
			word = word.lower()
			if(False == self.is_stop_word(word) and word.isnumeric()==False):
				s_word = stemmer.stem(word)

			#	s_word = word
			## it is not a stop word, check if the word
			## is already part of the article dictionary.
			## if yes, increment the count else add it.
			## If you are adding check if it is part of
			## the big corpus, if yes increment the count
			## of number of articles with that word.
				self.globalWordCount+=1
				new_art.doc_len = new_art.doc_len + 1
				if(s_word in article_dic):
					article_dic[s_word].wrd_count+=1
					global_dic[s_word].wrd_count+=1
				else:
					article_dic[s_word] = local_word_attributes(1)

					if (s_word in global_dic):
						global_dic[s_word].art_count+=1
						global_dic[s_word].wrd_count+=1
					else:
						global_dic[s_word] = global_word_attributes(1,1, 1, 0)




	def build_document_corpus(self):
		for raw_art in self.list_articles:
		#Create the dictionary for the given article and calculate term frequency.
			self.parse_raw_data(raw_art)
			for t in raw_art.topics:
				if t not in self.topicsMap:
					self.topicsMap[t]=[]
				self.topicsMap[t].append(raw_art)

			#raw_art.term_freq(self.globalWordCount)

		self.calcTfIdf()

	def writeVector(self,ftrain,ftest):
		for art in self.list_articles:#loop through all articles

			feature=art.featureVector
			if(not art.topics):
				f=ftest
				self.articleTest.append(art)
			else:
				f=ftrain
				self.articleTrain.append(art)
			"""
			for t in self.topicsList:
				if(t in art.topics):
					art.topicsMap[t]=1
				else:
					art.topicsMap[t]=0
			for p in self.placesList:
				if p in art.places:
					art.placesMap[t]=1
				else:
					art.placesMap[t]=0
			"""
			for t in art.topics:#print class labels
				#print(t,end=" ")
				f.write(t+",")


			#print(";",end="")
			f.write(";")

			for p in art.places:
				#print(p,end=" ")
				f.write(p+",")

			#print(art.id,end="")
			f.write(art.Id+"\t")




			for (word,value) in self.sortedDictionary.items():#for every word in the sorted dictionary. this defines the dimensions of the feature vector
				if(word in art.words):#if word in the dictionary exists in the article, only then does the vector for the article have a non zero dimension
					feature.append(art.words[word].wrd_count)


					f.write(str(art.words[word].wrd_count)+"\t")
					#print(str(art.words[word].term_fre)+" "+str(value.idf));
				else:
					feature.append(0)
					f.write(str(0)+"\t")
			f.write("\n")



	def writeVectorTfIdf(self,ftrain,ftest):
		for art in self.list_articles:#loop through all articles
			feature=art.tfidfFeatureVector
			if(not art.topics):
				f=ftest

			else:
				f=ftrain


			for t in art.topics:#print class labels
				#print(t,end=" ")
				f.write(t+",")


			#print(";",end="")
			f.write(";")

			for p in art.places:
				#print(p,end=" ")
				f.write(p+",")

			#print(art.id,end="")
			f.write(art.Id+"\t")

			for (word,value) in self.tfIdfDict.items():#for every word in the sorted dictionary. this defines the dimensions of the feature vector
				if(word in art.words):#if word in the dictionary exists in the article, only then does the vector for the article have a non zero dimension

					feature.append(value.tf_idf)
					f.write(str(value.tf_idf)+"\t")
					#print(str(art.words[word].term_fre)+" "+str(value.idf));
				else:
					feature.append(0)
					f.write(str(0)+"\t")
			f.write("\n")
	def filterWords(self):
		global_dic=self.raw_dictionary
		self.sortedDictionary=OrderedDict(sorted(global_dic.items(),key=lambda x:x[1].wrd_count))#returns a sorted global dictionary sorted by the frequencies
		self.tfIdfDict=OrderedDict(sorted(global_dic.items(),key=lambda x:x[1].tf_idf))
		length=len(self.sortedDictionary)*95.5/100## eliminate bottom and top 1%

		i=0
		for (key,value) in self.sortedDictionary.items():#loop to eliminate
			self.sortedDictionary.popitem(last=False)#delete from beginning

			i+=1
			if(i>=length):
				break
		self.sortedDictionary.popitem()
		i=0
		for (key,value) in self.tfIdfDict.items():
			self.tfIdfDict.popitem(last=False)#delete from beginning

			i+=1
			if(i>=length):
				break
		self.tfIdfDict.popitem(last=False)
		"""
		print("Length of feature vector(Number of dimensions i.e number of words)="+str(len(self.sortedDictionary)))#seems to be a bit high, we need to stem
		print("Words chosen (stemmed) followed by the count of each word:")
		for (key,value) in self.sortedDictionary.items():
			print(key+" "+str(value.wrd_count))
		print("Length of tf idf feature vector(Number of dimensions i.e number of words)="+str(len(self.tfIdfDict)))#seems to be a bit high, we need to stem
		print("Words chosen (stemmed) followed by the count of each word:")
		for (key,value) in self.tfIdfDict.items():
			print(key+" "+str(value.wrd_count))
		"""
		try:
			f=open('ftrain','w')

			f3=open('ftrainfidf','w')
			ftest=open('ftest','w')
			ftesttfidf=open('ftesttfidf','w')
			self.writeVector(f,ftest)
			self.writeVectorTfIdf(f3,ftesttfidf)

		except BaseException as e:
			print(e)

	def calcTfIdf(self):
        #Calculate IDF for the entire corpus.
        #IDF = log(Total number of documents / Number of documents with term t in it).
		number_of_documents = len(self.list_articles)
		# Number of elemets in the list gives the number of articles
		for key, value in self.raw_dictionary.items():
			value.idf = math.log(float(number_of_documents)/value.art_count)
			value.tf=float(value.wrd_count)/self.globalWordCount
			value.tf_idf=float(value.idf)*value.tf


class Article:
	def __init__(self):
		self.topics=[];
		self.places=[];
		self.body="";
		self.words = {}
		self.doc_len = 0
		self.featureVector=[]#stores final feature vector
		self.tfidfFeatureVector=[]#tf-idf feature vector
		self.Id=""
		#self.topicsMap=OrderedDict()
		#self.placesMap=OrderedDict()
		self.featureTest=[]
		self.tfIdfTest=[]

class MyHTMLParser(HTMLParser):
	def __init__(self):
		self.article=Article()
		HTMLParser.__init__(self)
		self.topicTag=0;
		self.placesTag=0;
		self.bodyTag=0;
		self.ListTag=0;
		self.articleList=[]
		self.topicsList=[]
		self.placesList=[]
		self.articleMap={}
	def handle_starttag(self, tag, attrs):

		if tag.upper()=="REUTERS":
			self.article=Article()
			for key,value in attrs:
				if key.lower()=="newid":
					self.article.Id=value

		if tag.upper()=="TOPICS":

			self.topicTag=1;
		elif tag.upper()=="BODY":
			self.bodyTag=1;
		elif tag.upper()=="PLACES":
			self.placesTag=1;
		elif tag.upper()=="D":
			self.ListTag=1

	def handle_endtag(self, tag):

		if tag.upper()=="TOPICS":
			self.topicTag=0;
		elif tag.upper()=="BODY":
			self.bodyTag=0;
			self.articleList.append(self.article)
		elif tag.upper()=="PLACES":
			self.placesTag=0;
		elif tag.upper()=="D":
			self.ListTag=0;
		elif tag.upper()=="REUTERS":
			self.articleMap[self.article.Id]=self.article

	def handle_data(self, data):
		if self.topicTag==1 and self.ListTag==1:
			if data not in self.topicsList:
				self.topicsList.append(data)
			self.article.topics.append(data);
		elif self.bodyTag==1:
			self.article.body=data;
		elif self.placesTag==1 and self.ListTag==1:
			if data not in self.placesList:
				self.placesList.append(data)
			self.article.places.append(data);
url = "http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-0"
parser=MyHTMLParser()
for i in range(1):
	if i<10:
		url1=url+"0"+str(i)+".sgm"

	else:
		url1=url+str(i)+".sgm"


	response=urllib.urlopen(url1)

	raw = response.read().decode('utf-8','replace')

	parser.feed(raw)

	"""
for article in parser.articleList:
	print("TOPICS:")
	for topic in article.topics:
		print(topic+"\n")
	print("PLACES:")
	for place in article.places:
		print (place+"\n")
	print(article.body)
"""

run = corpus()

run.get_raw_data(parser)
run.build_document_corpus()
run.filterWords()#This will filter the top 1% and bottom 1 % from the global dictionary based on frequencies
X=[]
Y=[]
X_test=[]
run.genTopicTestMap(X,Y,X_test)
run.pre=default_timer()-run.startClass
#run.classify(KNeighborsClassifier(n_neighbors=15),"KNN",X,Y,X_test)

#run.classify(tree.DecisionTreeClassifier(),"Decision Tree",X,Y,X_test)


##################################################################3


data_matrix = []
list_articles_with_topic = []
for art in run.list_articles:
    if art.topics != []:
        data_matrix.append(art.featureVector)
        list_articles_with_topic.append(art)

class kmeans:
    def __init__(self, list_articles, data_matrix, metric, num_cluster):
        self.data = data_matrix
        self.list_articles = []
        self.metric = metric
        self.num_clusters = num_cluster
        self.sets = []
        s = random.sample(xrange(len(self.data)), num_cluster)
        for i in range(len(s)):
            self.sets.append([s[i]])

#Precomputing similarity matrix
        self.similarity = []
        for art in list_articles:
            if art.topics != []:
                self.list_articles.append(art)

        for i in range(len(self.list_articles)):
            self.similarity.append([])

        for i in range(len(self.list_articles)):
            source = self.list_articles[i]
            self.similarity[i].append(1)
            for j in range(i+1, len(self.list_articles)):
                dest = self.list_articles[j]
                sim = self.eucledian(source.featureVector, dest.featureVector, metric)
                self.similarity[i].append(sim)
                self.similarity[j].append(sim)

    def eucledian(s, l1, l2, t):
        """
        if t == 1 eucledian
        t== 2 is jacard similarity
        """
        if(len(l1) != len(l2)):
            print("ERROR lenghts do not match");
            return 0
        if (t == 1):

            dis = 0.0;
            for i in range(0, len(l1)):
                x = l1[i] - l2[i]
                if(x != 0):
                    x = 1;
                dis = dis + x;

            return math.sqrt(dis)

        elif (t == 2):
            sim = 0.0
            uni = 0.0
            for i in range(len(l1)):
                if(l1[i] != 0 and l2[i] != 0):
                    sim = sim + 1

                if(l1[i] != 0 or l2[i] != 0):
                    uni = uni + 1;

            return(sim/uni)

        else:
            print("ERR:unknown metric")


    def run(self):
# for every articl find the distance between the centres and add it to the closest
        for itera in range(5):
            return_list = []
            for i in range(len(self.data)):
                mini = float("inf");
                index = -1;
                art = self.data[i]
                for j in range(len(self.sets)):
                    #Pick the first element in the set. i.e is our centroid.
                    if(self.sets[j][0] == i):
                        #we are checking centroid break
                        break
                    centroid = self.data[self.sets[j][0]]
                    dis = self.similarity[i][self.sets[j][0]]
                    if self.closer(dis, mini, self.metric):
                        mini = dis
                        index = j
                if(self.sets[j][0] == i):
                    continue
                    #we are checking centroid break

                self.sets[index].append(i)
                return_list.append(index)

            new_medians = self.find_new_medians()
            count = 0;
# If the medians to not change then they have converged, return.
            for k in range(len(new_medians)):
                if new_medians[k][0] == self.sets[k][0]:
                    count = count + 1
            if(count == len(new_medians)):
           #     print("Medians converged");
                break;
            self.sets = new_medians

        return return_list


    def check(self):
        """
        For each cluster find out the topics of each
        article and find out the matching ratio.
        """
        for clusters in self.sets:
            all_topics = set()
            topics_count = {}
            for i in clusters:
                article = self.list_articles[i]
                for topic in article.topics:
                    all_topics.add(topic)
            print("Cluster n topics ", all_topics)
            for topics in all_topics:
                topics_count[topics] = 0;
            for i in clusters:
                article = self.list_articles[i]
                for topic in article.topics:
                    for s in all_topics:
                        if s in topic:
                            topics_count[s] = topics_count[s] + 1
            print(topics_count)


    def closer(self, x , y, m):
        """
        is x closer than y is?
        if its distance, x < y
        if its similarity x > y
        """
        if(m == 1):
            if(x < y):
                return True
            else:
                return False
        else:
            if(y == float("inf")):
                y = -y
                return True
            if(x > y):
                return True
            else:
                return False


    def find_new_medians(self):
        new_medians = []
        for i in range(len(self.sets)):
            mini = float("inf");
            median = -1;
            for j in range(len(self.sets[i])):
                start = self.sets[i][j]
                dis = 0
                for k in range(len(self.sets[i])):
                    end = self.sets[i][k]
                    dis = dis + self.similarity[start][end]
                    # Find the article closest to all in the other articles in the current set.
                if self.closer(dis, mini, self.metric):
                    mini = dis
                    median = start

            new_medians.append([median])
        return new_medians



    def display(self):
        for i in range(len(self.sets)):
            print("Art" ,self.sets[i][0])
            print("Num of articles", len(self.sets[i]))
            for j in range(len(self.sets[i])):
                print(self.sets[i][j])


"""
kmclus.display()
kmclus.check()
"""


print("#####################################");
print("KMeans clustering, Euclidean");

start_time = default_timer()
kmclust = clust.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)

result = kmclust.fit_predict(data_matrix)
timeCluster = default_timer() - start_time
print("Time to cluster ", timeCluster)

entropy(result, list_articles_with_topic, 8);
"""
n_clusters=8
verify = []

cluster_table = []
for n in range(n_clusters):
    cluster_table.append([])

for art in run.list_articles:
    if art.topics != []:
        result = kmclust.predict(art.featureVector)
        for i in result:
            cluster_table[i].append(art)
            verify.append(i)


for i in range(len(cluster_table)):
#    print(i)
    for a in cluster_table[i]:
#        print(a.topics)
        pass

class heirarchial:
    def __init__(self, list_articles, data_matrix, metric, num_cluster):
        self.list_articles = list_articles
        self.data_matrix = data_matrix
        self.metric = metric
        self.num_cluster = num_cluster

    def run(self):
        if(self.metric == 1):
            data_matrix = self.data_matrix
            distance_matrix = sklearn.metrics.pairwise.euclidean_distances(data_matrix, data_matrix)
            # Create n sets each with single element
            n_sets = []
            minimums = []
            for i in range(len(distance_matrix)):
                n_sets.append(i)
                minimums.append(float("inf"))

            m = float("inf")
            for i in range(0, len(distance_matrix)):
                for j in range(i+1,len(distance_matrix)):
                    if(distance_matrix[i][j] < m):
                        m = minimums[i] = distance_matrix[i][j];

            for k in range(len(distance_matrix) - self.num_cluster):
                minimum = 99999
                min_i = -1;
                min_j = -1;
                for i in range(0, len(distance_matrix)):
                    for j in range(i+1,len(distance_matrix)):
                        if(distance_matrix[i][j] <= minimum and n_sets[i] != n_sets[j]):
                            min_i = i;
                            min_j = j;
                            minimum = distance_matrix[i][j]
                distance_matrix[min_i][min_j] = float("inf")
                self.union(n_sets,n_sets[min_i],n_sets[min_j]);

            print(n_sets)




    def union(self, array, i, j):
        if i < j:
            key = i
        else:
            key = j

        for x in range(len(array)):
            if (array[x] == i or array[x] == j):
                array[x] = key

print("#####################################");
print("Agglomerative clustering, cosine");

start_time = default_timer()
heir = sklearn.cluster.AgglomerativeClustering(n_clusters=8, affinity='cosine', connectivity=None, n_components=None, compute_full_tree='auto',linkage='complete')

X =[]
for art in run.list_articles:
    if art.topics != []:
        X.append(art.featureVector)

X = np.array(X)

result = heir.fit_predict(X);
timeCluster = default_timer() - start_time
print("Time to cluster ", timeCluster)

entropy(result, list_articles_with_topic, 8);



print("#####################################");
print("Agglomerative clustering, manhattan");

start_time = default_timer()
heir = sklearn.cluster.AgglomerativeClustering(n_clusters=8, affinity='manhattan', connectivity=None, n_components=None, compute_full_tree='auto',linkage='complete')

X =[]
for art in run.list_articles:
    if art.topics != []:
        X.append(art.featureVector)

X = np.array(X)

result = heir.fit_predict(X);
timeCluster = default_timer() - start_time
print("Time to cluster ", timeCluster)

entropy(result, list_articles_with_topic, 8);




print("#####################################");
print("Agglomerative clustering, Euclidean");

start_time = default_timer()
heir = sklearn.cluster.AgglomerativeClustering(n_clusters=8, affinity='euclidean', connectivity=None, n_components=None, compute_full_tree='auto',linkage='ward')

X =[]
for art in run.list_articles:
    if art.topics != []:
        X.append(art.featureVector)

X = np.array(X)

result = heir.fit_predict(X);

timeCluster = default_timer() - start_time
print("Time to cluster ", timeCluster)
entropy(result, list_articles_with_topic, 8);

"""
