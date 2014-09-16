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
			if(!is_stop_word(word)):
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
				 	if(global_dic.has_key(s_word)):
						temp_word_global = global_dic[s_word]
						temp_word_global.art_count = temp_word_global.art_count + 1
					else:
					 	new_global_attr = global_word_attributes(1, 0)
					 	global_dic[s_word] = new_global_attr
					 	
					 	
				


	def build_document_corpus(self):
		for raw_art in self.list_raw_articles:
			self.parse_raw_data(raw_art)
		    





