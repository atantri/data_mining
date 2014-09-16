import urllib.request
import nltk
from nltk.corpus import stopwords
from html.parser import HTMLParser
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer

class global_word_attributes:
	"""
	Count here holds the number of articles with
	the presence.
	"""
	def __init__(self, count, tf_idf):
		self.art_count = count
		self.tf_idf = tf_idf

class local_word_attributes:
	"""
	Count here holds the number of times the word occurs
	in the article
	"""
	def __init__(self, count):
		self.wrd_count = count

		
class corpus:
	def __init__():
		self.list_raw_articles = []
		self.raw_dictionary = {}
        
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


	def get_raw_data(self):
		self.list_raw_articles = call_aneesh();

	def parse_raw_data(self, new_art):
		tokenizer = RegexpTokenizer(r'\w+')
		tokens = tokenizer.tokenize(new_art.data)
		stemmer = LancasterStemmer()
		article_dic = new_art.words
		global_dic = self.raw_dictionary
		
		for word in tokens:
			word = word.lower()
			if(False == is_stop_word(word)):
				s_word = stemmer(word)
			## it is not a stop word, check if the word
			## is already part of the article dictionary.
			## if yes, increment the count else add it.
			## If you are adding check if it is part of 
			## the big corpus, if yes increment the count
			## of number of articles with that word.
				new_art.doc_lenght = new_art.doc_lenght + 1
				if(article_dic.has_key(s_word)):
					temp_word_attr = article_dic[s_word]
					temp_word_attr.wrd_count = temp_word_attr.wrd_count + 1
				else:
					new_art_attr = local_word_attributes(1)
					article_dic[s_word] = new_art_attr
					if global_dic.has_key(s_word):
						temp_word_global = global_dic[s_word]
						temp_word_global.art_count = temp_word_global.art_count + 1
					else:
						new_global_attr = global_word_attributes(1, 0)
						global_dic[s_word] = new_global_attr
							
	def build_document_corpus(self):
		for raw_art in self.list_raw_articles:
			self.parse_raw_data(raw_art)
		    





class Article:	
	def __init__(self):
		self.topics=[];
		self.places=[];
		self.body="";
		self.words = {}
		self.doc_len = 0

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
for i in range(1):
	if i<10:
		url1=url+"0"+str(i)+".sgm"
		print(url+"0"+str(i)+".sgm")
	else:
		url1=url+"0"+str(i)+".sgm"
		print(url+str(i)+".sgm")

	raw = urllib.request.urlopen(url1).read().decode('utf8')
	
	parser.feed(raw)
	print(len(parser.articleList))
for article in parser.articleList:
	print("TOPICS:")
	for topic in article.topics:
		print(topic+"\n")
	print("PLACES:")
	for place in article.places:
		print (place+"\n")

	print(article.body)
