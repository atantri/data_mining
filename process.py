import urllib
import nltk
import math
from nltk.corpus import stopwords
from HTMLParser import HTMLParser
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
import operator
import collections
from collections import OrderedDict
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


	def get_raw_data(self, lis):
		self.list_articles = lis;

	def parse_raw_data(self, new_art):
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
			#raw_art.term_freq(self.globalWordCount)
		
		self.calcTfIdf()
		

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
		self.tfIdfDict.popitem()
		print("Length of feature vector(Number of dimensions i.e number of words)="+str(len(self.sortedDictionary)))#seems to be a bit high, we need to stem
		print("Words chosen (stemmed) followed by the count of each word:")
		for (key,value) in self.sortedDictionary.items():
			print(key+" "+str(value.wrd_count))
		print("Length of tf idf feature vector(Number of dimensions i.e number of words)="+str(len(self.tfIdfDict)))#seems to be a bit high, we need to stem
		print("Words chosen (stemmed) followed by the count of each word:")
		for (key,value) in self.tfIdfDict.items():
			print(key+" "+str(value.wrd_count))
		try:
			f=open('featureVector','w')
			f2=open('featureVectorSimple','w')
			f3=open('featureVectortfidf','w')
			print("a");
			for art in self.list_articles:#loop through all articles
				
				for t in art.topics:#print class labels
					#print(t,end=" ")
					f.write(t+",")
					f2.write(t+",")
					f3.write(t+",")
				#print(";",end="")
				f.write(";")
				f2.write(";")
				f3.write(";")
				for p in art.places:
					#print(p,end=" ")
					f.write(p+",")
					f2.write(p+",")
					f3.write(p+",")
				#print(art.id,end="")
				f.write(art.id+" ")
				f2.write(art.id+" ")
				f3.write(art.id+" ")
				for (word,value) in self.sortedDictionary.items():#for every word in the sorted dictionary. this defines the dimensions of the feature vector
					if(word in art.words):#if word in the dictionary exists in the article, only then does the vector for the article have a non zero dimension
						art.featureVector.append(art.words[word].wrd_count)
						f.write(str(art.words[word].wrd_count)+" ")
						#print(str(art.words[word].term_fre)+" "+str(value.idf));
					else:
						art.featureVector.append(0)
						f.write(str(0)+" ")
				
				for (word,value) in self.tfIdfDict.items():#for every word in the sorted dictionary. this defines the dimensions of the feature vector
					if(word in art.words):#if word in the dictionary exists in the article, only then does the vector for the article have a non zero dimension
						
						art.tfidfFeatureVector.append(value.tf_idf)
						f3.write(str(value.tf_idf)+" ")
						#print(str(art.words[word].term_fre)+" "+str(value.idf));
					else:
						art.tfidfFeatureVector.append(0)
						f3.write(str(0)+" ")
				#for dim in art.featureVector:
				#	f.write(str(dim)+" ")
				for word in art.words:
					f2.write(word+":"+str(art.words[word].wrd_count)+" ")
				#for dim in art.tfidfFeatureVector:
				#	f3.write(str(dim)+" ")
				#print("")
				f.write("\n\n")
				f2.write("\n\n")
				f3.write("\n\n")
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
		self.id=""
			


class MyHTMLParser(HTMLParser):
	def __init__(self):
		self.article=Article()
		HTMLParser.__init__(self)
		self.topicTag=0;
		self.placesTag=0;
		self.bodyTag=0;
		self.ListTag=0;
		self.articleList=[]
	def handle_starttag(self, tag, attrs):
		
		if tag.upper()=="REUTERS":
			for key,value in attrs:
				if key.lower()=="oldid":
					self.article.id=value

		if tag.upper()=="TOPICS":
			self.article=Article()
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

	def handle_data(self, data):
		if self.topicTag==1 and self.ListTag==1:
			
			self.article.topics.append(data);
		elif self.bodyTag==1:
			self.article.body=data;
		elif self.placesTag==1 and self.ListTag==1:
			self.article.places.append(data);
url = "http://web.cse.ohio-state.edu/~srini/674/public/reuters/reut2-0"
parser=MyHTMLParser()
for i in range(22):
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

run.get_raw_data(parser.articleList)
run.build_document_corpus()
run.filterWords()#This will filter the top 1% and bottom 1 % from the global dictionary based on frequencies
"""
for (word,obj) in run.sortedDictionary.items():
	print(word+" "+str(obj.wrd_count));

	
for entry in run.list_articles:
	pass
#	print(entry.topics)
"""
