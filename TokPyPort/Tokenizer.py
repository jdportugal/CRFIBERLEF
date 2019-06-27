import nltk.data
import os
from nltk.corpus import floresta
import nltk
import xmltodict


def load_token_configurations(config_file):
	contractions_path = ""
	clitics_path = ""
	with open(config_file) as g:
		for line in g:
			if(line[0]!="#"):
				if(line.split("=")[0]=="contractions"):
					contractions_path = line.split("=")[1].strip('\n')
				elif(line.split("=")[0]=="clitics"):
					clitics_path = line.split("=")[1].strip('\n')
	return contractions_path,clitics_path

def get_input_from_file(fileinput):
	text = " "
	with open(fileinput,'r') as f:
		for line in f:
			if("#INPUT:" not in line and "#FORMAT:" not in line):
				text += line.strip('\n') 
				if(len(text)>0):
					text+=" " 
	return text

def replace_contrations(contractions_path,tokens):
	tokens_after_contractions = []
	encontrou = 0

	#Check if tokens contain contractions
	#If so, change them to the most extended form
	with open(contractions_path) as fd:
		doc = xmltodict.parse(fd.read())
		result = (doc["contractions"]["replacement"])
		for tok in tokens:
			encontrou = 0
			token2 = tok
			for elem in result:
				if(tok==elem['@target']):
					encontrou=1
					subs = elem['#text'].split(" ")
					for part in subs:
						tokens_after_contractions.append(part)
			#if word in not contration add it as it was
			if(encontrou==0):
				tokens_after_contractions.append(token2)
	return tokens_after_contractions

def replace_clitics(clitics_path,tokens):
	tokens_after_clitics =[]
	with open(clitics_path) as fd:
		doc2 = xmltodict.parse(fd.read())
		result2 = (doc2["clitics"]["replacement"])
		for tok2 in tokens:
			if(len(tokens)>0):
				encontrou = 0
				token2 = tok2
				for elem2 in result2:
					if(tok2==elem2['@target']):
						encontrou=1
						subs = elem2['#text'].split(" ")
						for part in subs:
							tokens_after_clitics.append(part)
				if(encontrou==0):
					withslash = tok2.split("-")
					if(len(withslash)>1):
						nova_palavra = ""
						for parte in withslash:
							if(parte!=withslash[0]):
								nova_palavra +="-" + parte
						encontrou = 0
						token2 = tok2
						for elem2 in result2:
							if(nova_palavra==elem2['@target']):
								encontrou=1
								subs = elem2['#text'].split(" ")
								for part in subs:
									tokens_after_clitics.append(part)
						#if word in not contration add it as it was
						if(encontrou==1):
							tokens_after_clitics.append(withslash[0]+"-")
				if(encontrou==0):
					tokens_after_clitics.append(token2)
	return tokens_after_clitics	

def nlpyport_tokenizer(fileinput,TokPort_config_file):
	#define the tagset being used
	text = " "
	text = get_input_from_file(fileinput).split(" ")

	return text

'''
if __name__ == '__main__':
	print(nlpyport_tokenizer("EntradaCadeiaTotal.txt"))
'''