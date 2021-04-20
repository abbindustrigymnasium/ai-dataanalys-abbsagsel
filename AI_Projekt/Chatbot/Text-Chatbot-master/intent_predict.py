from dependency_tree import to_nltk_tree , spacy_desc_parser
import spacy
from preprocessor import nlp
import numpy as np
from preprocessor import pad_sequences, labels
from keras.models import load_model

classes = len(labels)


model = load_model('nishank/intent_models/smart_model_test2.h5') #load the model to be tested.

sen_test = int(input("\nNumber of test sent? \n"))
sen_test_vec_seq = [] #list of all vectorized test sent
sen_test_ent_seq = [] #list of lists of entities in each test sent
sen_test_seq = [] #list of all test sent
for i in range(sen_test):
	print ("Enter Sent",i+1," to be classified:")
	test_text = input()
	sen_test_seq.append(test_text)
	#vectorize text.
	sen_doc = nlp(test_text)
	sent_vec_test = []
	for word in sen_doc:
		sent_vec_test.append(word.vector)
	sen_test_vec_seq.append(sent_vec_test)
	sen_test_ent_seq.append(sen_doc.ents)
	
#convert all the sentences into matrices of equal size.
sen_test_vec_seq = pad_sequences(sen_test_vec_seq)

#get predictions
prediction = model.predict(sen_test_vec_seq)

classl_predictions = np.zeros(prediction.shape)
'''
 classl_predictions[i]  intent predictions for Sent i, where:
 One of the following integer values is saved per intent label:
 2 - This is the most probable intent of the given Sent
 1 - This could possibly be a sub intent of the given Sent
 0 - This intent is not present in the given Sent 
'''
for i in range(sen_test):
	m = max(prediction[i])
	p = np.where(prediction[i] > 0.55 * m)	# p collects possible sub intents
	q = np.where(prediction[i] == m)	#q collects intent
	classl_predictions[i][p] = 1
	classl_predictions[i][q] = 2



for i in range(sen_test):
	print("\n\t SentDisplaying  Predictions:")
	print(" Sent ", i+1, " :", sen_test_seq[i])
	print(" Entities Recognized:", end = "\t")
	if len(sen_test_ent_seq[i]) == 0:
		print(" None.", end = "\t")
	for ent in sen_test_ent_seq[i]:
		print(" ",ent.label_, ent.text, end= "\t")
	print("\n Dependency tree:")
	tx = nlp(sen_test_seq[i])
	[to_nltk_tree(sent.root).pretty_print() for sent in tx.sents]
	#spacy_desc_parser(tx)	#Prints subject - activity - numbers
	#Using the to spacy desc function is more useful in general statements. 
	#I will include this as a functionality once there is enough data to incorporate such a class.
	for x in range(len(classl_predictions[i])):
		if classl_predictions[i][x] == 2 :
			print(" Detected intent: ",labels[x])
		if classl_predictions[i][x] == 1:
			print(" Detected possible sub-intent: ",labels[x])
	if  len(set(classl_predictions[i])) == 1:
		print(" Could not detect intent. ")
		
print("\nTest Complete")
