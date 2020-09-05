import pickle
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
class PredictSentiment:
	def __init__(self):
		self.model = None
		self.vectorizer = None
		with open("model_old", 'rb') as data:#logistic regression deployment  
			self.model = pickle.load(data)
		with open("vectorizer_old", 'rb') as data:
			self.vectorizer = pickle.load(data)

	def lemma(self,text):
		w_tokenizer = TweetTokenizer()
		lemmatizer = nltk.stem.WordNetLemmatizer()
		return " ".join([lemmatizer.lemmatize(w) for w  in w_tokenizer.tokenize(text)])

	def input(self,review_text):

		sample_dict = {"Processed_Phrase":[review_text]}
		sample_df = pd.DataFrame(sample_dict)

		sample_df["Processed_Phrase"] = sample_df["Processed_Phrase"].str.lower()
		sample_df["Processed_Phrase"] = sample_df.Processed_Phrase.apply(self.lemma)


		x_test = self.vectorizer.texts_to_sequences(sample_df["Processed_Phrase"])


		
		y_test_pred=self.model.predict_classes(x_test)
		final_pred = y_test_pred[0]

		if final_pred == 1:
			return "IT BELONGS TO CLASS A"
		elif final_pred == 2:
			return "IT BELONGS TO CLASS B"
		elif final_pred == 3:
			return "IT BELONGS TO CLASS C"
		elif final_pred == 4:
			return "IT BELONGS TO CLASS D"
		else:
			return "IT DOES NOT BELONG TO ANY CLASS"


if __name__ == "__main__" :
	pred = PredictSentiment()
	print(pred.input("posts"))


