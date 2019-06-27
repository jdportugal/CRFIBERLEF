PARTICIPANTS: João Ferreira, Hugo Gonçalo Oliveira, Ricardo Rodrigues


INSTITUTE: University of Coimbra

---------------------------------------

DATASETS:
	TRAIN: SECOND HAREM (1)(2)
	TEST: SECOND HAREM (1)(2)

	(1) Freitas, Cláudia, et al. "Second HAREM: advancing the state of the art of named entity recognition in Portuguese." quot; In Nicoletta Calzolari; Khalid Choukri; Bente Maegaard; Joseph Mariani; Jan Odijk; Stelios Piperidis; Mike Rosner; Daniel Tapias (ed) Proceedings of the International Conference on Language Resources and Evaluation (LREC 2010)(Valletta 17-23 May de 2010) European Language Resources Association. European Language Resources Association, 2010.
	(2) As formated by André Pires : https://github.com/arop/ner-re-pt
---------------------------------------

LIBRARIES:

nltk==3.4.1
numpy==1.16.2
pandas==0.24.2
python-crfsuite==0.9.6
python-dateutil==2.8.0
pytz==2019.1
scikit-learn==0.20.3
scipy==1.2.1
singledispatch==3.4.0.3
six==1.12.0
sklearn==0.0
sklearn-crfsuite==0.3.6
tabulate==0.8.3
tqdm==4.31.1
xmltodict==0.12.0

---------------------------------------
INSTRUCTIONS:
---------------------------------------

STEP 1: Using pip, install all the necessary libraries

	pip install -r requirements.txt

STEP 2: In the NLPyPort directory, run in the terminal:

	python FullPipeline.py input_file.txt output_file.txt
	(eg: python FullPipeline.py Task1-Expected-Input.txt out.txt)

	If you wish to test more recent versions simply add one of the following after the other arguments
	- sigarra (eg: python FullPipeline.py Task1-Expected-Input.txt out.txt sigarra)
	- sigarra_harem (eg: python FullPipeline.py Task1-Expected-Input.txt out.txt sigarra_harem)