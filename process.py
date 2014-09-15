import urllib.request
import nltk
from nltk.corpus import stopwords
from html.parser import HTMLParser
class Article:	
	def __init__(self):
		self.topics=[];
		self.places=[];
		self.body="";
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
