import nltk.data
import os
from nltk.corpus import floresta
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus.reader import TaggedCorpusReader
from nltk.tokenize import LineTokenizer
from nltk.corpus import treebank
from nltk.metrics import accuracy
from nltk.corpus import machado
import pickle
import nltk
import time
import xmltodict
from LemPyPort.LemFunctions import *
from LemPyPort.dictionary import *
from TokPyPort.Tokenizer import *
from TagPyPort.Tagger import *
from CRF.CRF_Teste import *
import sys
import os


global_porperties_file = "config/global.properties"

lexical_conversions="PRP:PREP;PRON:PRO;IN:INTERJ;ART:DET;"
floresta.tagged_words(tagset = "pt-bosque")
TokPort_config_file = ""
TagPort_config_file = ""
LemPort_config_file = ""

def load_config(config_file="config/global.properties"):
	global TokPort_config_file
	global TagPort_config_file
	global LemPort_config_file
	with open (config_file,'r') as f:
		for line in f:
			if(line[0]!="#"):
				if(line.split("=")[0]=="TokPort_config_file"):
					TokPort_config_file = line.split("=")[1].strip("\n")
				elif(line.split("=")[0]=="TagPort_config_file"):
					TagPort_config_file = line.split("=")[1].strip("\n")
				elif(line.split("=")[0]=="LemPort_config_file"):
					LemPort_config_file = line.split("=")[1].strip("\n")
# """word tokenizer"""
def tokenize(fileinput):
	return nlpyport_tokenizer(fileinput,TokPort_config_file)

def tag(tokens):
	return nlpyport_pos(tokens,TagPort_config_file)


def lematizador_normal(tokens,tags):
	global LemPort_config_file
	mesmas = 0
	alteradas = 0
	resultado = nlpyport_lematizer(tokens,tags,LemPort_config_file)
	return resultado

def load_manual(file):
	tokens = []
	tags = []
	f =  open(file,'r')
	alteradas = 0
	mesmas = 0
	for line in f:
		res = line.split(" ")
		if(len(res)>1):
			tokens.append(res[0])
			tags.append(res[1].split('\n')[0])
	return tokens,tags

def write_lemmas_only_text(lem,file="testes.txt"):
	for elem in lem:
		with open(file,'a') as f:
			if(elem == "#"):
				f.write('\n')
			else:
				f.write(str(elem)+" ")

def write_simple_connl(tokens,tags,lems,file=""):
	linhas = 0
	if(file != ""):
		for index in range(len(tokens)):
			with open(file,'a') as f:
				if(tokens[index] == "#"):
					f.write("\n")
					linhas = 0
				else:
					linhas += 1
					f.write(str(linhas) + ", " + str(tokens[index] + ", " +str(lems[index] + ", " + str(tags[index]))+"\n"))
	else:
		for index in range(len(tokens)):
			if(tokens[index] == "#"):
				#print("\n")
				linhas = 0
			else:
				linhas += 1
				#print(str(linhas) + ", " + str(tokens[index] + ", " +str(lems[index] + ", " + str(tags[index]))))


def lem_file(out,token,tag):
	lem = []
	ent = []
	lem = lematizador_normal(token,tag)
	with open(out,"wb") as f:
		for i in range(len(token)):
			line = token[i] +"\t" +tag[i] + "\t" + lem[i]  + "\n"
			f.write((line).encode('utf8'))
	
def join_data(tokens,tags,lem):
	data = []
	for i in range(len(tokens)):
		dados = []
		dados.append(tokens[i]) 
		dados.append(tags[i]) 
		dados.append(lem[i]) 
		data.append(dados)
	return data

def full_pipeline(infile,outfile,model="harem"):
	load_config()
	#############
	#Tokenize
	#############
	tokens = tokenize(infile)
	#############
	#Pos
	#############
	
	tags,result_tags = tag(tokens)
	for index,elem in enumerate(tags):
		if(tokens[index]==""):
			tokens[index]=" "
			tags[index]=" "
	
	for index,elem in enumerate(tags):
		if(":" in tags[index]):
			tags3 = tags[index].split(":")
			tags[index] = tags3[1]

	#############
	#Lemm
	#############
	lemas = lematizador_normal(tokens,tags)

	#############
	#Ent
	#############
	all_tokens = []
	joined_data = join_data(tokens,tags,lemas)
	trained_model = "CRF/trainedModels/"+model+".pickle"
	ents = run_crf(joined_data,trained_model)
	with open(outfile,"w") as f:
		f.write("#OUTPUT:" + "\n"+"#FORMAT: token predict_tag")
	for i in range(len(tokens)):
		#if(tokens[i]!=" "):
			#print(tokens[i] + " " + ents[i])
		if(tokens[i]==""):
			ents[i] = ""
		if(ents[i]=="B-TEMPO" or ents[i]=="B-Hora" or ents[i]=="B-TME" or ents[i]=="B-Data"):
			ents[i]="B-TME"
		elif(ents[i]=="I-TEMPO" or ents[i]=="I-Hora" or ents[i]=="I-TME" or ents[i]=="I-Data"):
			ents[i]="I-TME"
		elif(ents[i]=="B-VALOR" or ents[i]=="B-VALOR" or ents[i]=="B-VAL"):
			ents[i]="B-VAL"
		elif(ents[i]=="I-VALOR" or ents[i]=="I-VALOR" or ents[i]=="I-VAL"):
			ents[i]="I-VAL"	
		elif(ents[i]=="B-LOCAL" or ents[i]=="B-Localizacao" or ents[i]=="B-PLC"):
			ents[i]="B-PLC"
		elif(ents[i]=="I-LOCAL" or ents[i]=="I-Localizacao" or ents[i]=="I-PLC"):
			ents[i]="I-PLC"	
		elif(ents[i]=="B-PESSOA" or ents[i]=="B-Pessoa" or ents[i]=="B-PER"):
			ents[i]="B-PER"	
		elif(ents[i]=="I-PESSOA" or ents[i]=="I-Pessoa" or ents[i]=="I-PER"):
			ents[i]="I-PER"	
		elif(ents[i]=="B-ORGANIZACAO" or ents[i]=="B-Organizacao" or ents[i]=="B-ORG"):
			ents[i]="B-ORG"	
		elif(ents[i]=="I-ORGANIZACAO" or ents[i]=="I-Organizacao" or ents[i]=="I-ORG"):
			ents[i]="I-ORG"	
		else:
			ents[i]="O"
		if(tokens[i]!=" " and tokens[i]!='\n'):
			all_tokens.append(tokens[i] + " " + ents[i]+"\n")
			#print(tokens[i] + " " + ents[i])
		else:
			if(tokens[i]!="\n"):
				all_tokens.append("\n")
			#print(tokens[i])

	with open(outfile,"a") as f:
		for i in range(len(all_tokens)-1):
			if(i == len(all_tokens)-2):
				f.write(all_tokens[i].strip("\n"))
			else:
				f.write(all_tokens[i])
	#############


if __name__ == "__main__":
	#start_time = time.time()
	#"Task1-Examples/Task1-Expected-Input.txt"
	model=""
	if(len(sys.argv)==2):
		in_file = sys.argv[1]
		out_file = "Task-Out.txt"
	elif(len(sys.argv)==3):
		in_file = sys.argv[1]
		out_file = sys.argv[2]
	elif(len(sys.argv)>3):
		in_file = sys.argv[1]
		out_file = sys.argv[2]
		model = sys.argv[3]
	else:
		in_file = input("What is the name of the input file?\n")
		out_file = input("What is the name of the output file?\n")

	if(os.path.exists(in_file)):
		if(model!=""):
			full_pipeline(in_file,out_file,model)
		else:
			full_pipeline(in_file,out_file)
	else:
		print("There is no file with the given name in the current directory")
	
	#print("--- %s Seconds ---" % (time.time() - start_time))

