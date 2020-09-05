import pandas as pd
import cherrypy
import pickle
import numpy as np
from numpy import nan
import seaborn as sns  
import matplotlib.pyplot as plt 
import nltk
from wordcloud import WordCloud,STOPWORDS
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk .tokenize import sent_tokenize
from nltk import word_tokenize
col_names=["Type","Posts"]
data=pd.read_csv("C:\\Users\\Saarthu\\Downloads\\project.csv",names=col_names,engine='python',sep=',',quotechar='"',error_bad_lines=False)
print(data)
fa=data[data.Type=='A']
print(fa)
fb=data[data.Type=='B']
print(fb)
fc=data[data.Type=='C']
print(fc)
fd=data[data.Type=='D']
print(fd)
lowwords=set(ENGLISH_STOP_WORDS)
print(lowwords)
wordcloudposts=WordCloud(background_color='black',stopwords=lowwords,width=2000,height=1500).generate(str(fa['Posts']))
print(wordcloudposts)
plt.imshow(wordcloudposts)
plt.xlabel('TYPE A ')
plt.show()
wordcloudposts=WordCloud(background_color='black',stopwords=lowwords,width=2000,height=1500).generate(str(fb['Posts']))
print(wordcloudposts)
plt.imshow(wordcloudposts)
plt.xlabel('TYPE B ')
plt.show()
wordcloudposts=WordCloud(background_color='black',stopwords=lowwords,width=2000,height=1500).generate(str(fc['Posts']))
print(wordcloudposts)
plt.imshow(wordcloudposts)
plt.xlabel('TYPE C ')
plt.show()
wordcloudposts=WordCloud(background_color='black',stopwords=lowwords,width=2000,height=1500).generate(str(fd['Posts']))
print(wordcloudposts)
plt.imshow(wordcloudposts)
plt.xlabel('TYPE D ')
plt.show()
from cleantext import clean
y=data.iloc[:,0]
print(y)
print(y.shape) 
x=data.iloc[:,1]
print(x)
(clean(x,fix_unicode=True,lower=True,no_line_breaks=True,no_urls=True,no_emails=True,no_phone_numbers=True,no_numbers=True,no_digits=True,no_currency_symbols=True,no_punct=True))                                                                                                                    
print(x.shape)
print(data.isnull().sum(axis=0))
ax,fig = plt.subplots(figsize=(10, 7))
data_class = data["Type"].value_counts()
plt.hist(data_class,bins=None)
plt.show()
data_class.plot(kind= 'bar', color= ["blue", "orange"])
plt.title('Bar chart')
plt.show()
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
count=CountVectorizer(stop_words=lowwords)
vectorizer=TfidfVectorizer(stop_words=lowwords) 
with open("vectorizer","wb") as output:
 pickle.dump(vectorizer,output)
print(vectorizer.fit(x))
##print(vectorizer.build_preprocessor())
print(vectorizer.build_analyzer())
##print(vectorizer.build_tokenizer())
stw=(vectorizer.get_stop_words())
x=[word for word in x if word not in stw]
(vectorizer.get_feature_names())
x=vectorizer.transform(x)
print(x)
from sklearn import preprocessing
le=preprocessing.LabelEncoder()
print(le.fit(y))
print(list(le.classes_))
y=(le.transform(y))
print(y)
y=np.array(y).reshape(-1,1)
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=50)
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=50)
xtrain,ytrain=sm.fit_sample(xtrain,ytrain.ravel())
from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression()
with open("model","wb") as output:
 pickle.dump(classifier,output)
print(classifier.fit(xtrain,ytrain))
ypred=(classifier.predict(xtest))
ypred1=classifier.predict(xtrain)
print(ypred)
print((list(le.inverse_transform(ypred))))
print(classifier.predict_proba(xtest))
print(classifier.predict_log_proba(xtest))
print(classifier.decision_function(xtest))
print(classifier.get_params(deep=True))
print(classifier.score(xtrain,ytrain))
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,classification_report,f1_score
print(classification_report(ytest,ypred))
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
import math
rmse=math.sqrt(mean_squared_error(ytest,ypred))
print(rmse)
print(r2_score(ytest,ypred))
plt.plot(ypred,ytest)
plt.show()
plt.plot(ypred)
plt.show()
from sklearn.metrics import confusion_matrix
confusionmatrix=confusion_matrix(ypred,ytest)
print(confusionmatrix)
from sklearn.tree import DecisionTreeClassifier
tree=DecisionTreeClassifier(criterion='entropy')
print(tree.fit(xtrain,ytrain))
ypred=tree.predict(xtest)
ypred1=tree.predict(xtrain)
print(ypred)
print(list(le.inverse_transform(ypred)))
print(tree.apply(xtest))
print(tree.cost_complexity_pruning_path(xtrain,ytrain))
print(tree.decision_path(xtest))
print(tree.get_depth())
print(tree.get_n_leaves())
print(tree.get_params(deep=True))
print(tree.predict_log_proba(xtest))
print(tree.predict_proba(xtest))
print(tree.score(xtrain,ytrain))
from sklearn.model_selection import GridSearchCV
rmse=math.sqrt(mean_squared_error(ytest,ypred))
print(rmse)
print(r2_score(ytest,ypred))
print(classification_report(ytest,ypred))
confusionmatrix=confusion_matrix(ypred,ytest)
print(confusionmatrix)
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
plt.plot(ypred)
plt.show()
from sklearn.neighbors import KNeighborsClassifier 
knc=KNeighborsClassifier(n_neighbors=100)
print(knc.fit(xtrain,ytrain))
ypred=(knc.predict(xtest))
ypred1=knc.predict(xtrain)
print(ypred)
print(list(le.inverse_transform(ypred)))
print(knc.predict_proba(xtest))
print(knc.score(xtrain,ytrain))
print(knc.kneighbors())
print(knc.kneighbors_graph())
print(r2_score(ytest,ypred))
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import NeighborhoodComponentsAnalysis
nca=NeighborhoodComponentsAnalysis(random_state=42)
from mlxtend.preprocessing import DenseTransformer
nca_pipe=(make_pipeline((NeighborhoodComponentsAnalysis()),(KNeighborsClassifier())))
print(nca_pipe)
dense=DenseTransformer()
print(dense.fit(xtrain,ytrain))
##xtrain,ytrain=dense.transform(xtrain,ytrain)
##print(nca.fit(xtrain,ytrain))
##knc.fit(nca.transform(xtrain,ytrain))
##print(knc.score(nca.transform(xtest,ytest))
##print(nca_pipe.fit(xtrain,ytrain))
##print(nca_pipe.score(xtrain,ytrain))
print(classification_report(ytest,ypred))
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
confusionmatrix=confusion_matrix(ypred,ytest)
print(confusionmatrix)
rmse=math.sqrt(mean_squared_error(ypred,ytest))
print(rmse)
plt.plot(ypred)
plt.show()
from sklearn.ensemble import AdaBoostClassifier
adc=AdaBoostClassifier(random_state=0,learning_rate=1.0)
print(adc.fit(xtrain,ytrain))
ypred=adc.predict(xtest)
ypred1=adc.predict(xtrain)
print(ypred)
print(list(le.inverse_transform(ypred)))
print(classification_report(ypred,ytest))
print(adc.predict_proba(xtest))
print(adc.predict_log_proba(xtest))
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
import xgboost as xgb
import lightgbm as lgb 
from xgboost import plot_importance 
xgb1=xgb.XGBClassifier(booster='gbtree',n_jobs=-1,n_estimators=500,max_depth=0,learning_rate=0.3,random_state=14,max_leaves=5,grow_policy="lossguide")
print(xgb1.fit(xtrain,ytrain))
ypred=xgb1.predict(xtest)
ypred1=xgb1.predict(xtrain)
print(ypred)
print(xgb1.predict_proba(xtest))
print(list(le.inverse_transform(ypred)))
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
lgb1=lgb.LGBMClassifier(boosting_type="gbdt",num_leaves=5,n_estimators=500,n_jobs=-1,learning_rate=0.3,max_depth=0,random_state=14)
print(lgb1.fit(xtrain,ytrain))
ypred=lgb1.predict(xtest)
ypred1=lgb1.predict(xtrain)
print(ypred)
print(list(le.inverse_transform(ypred)))
print(lgb1.predict_proba(xtest))
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
rf=RandomForestClassifier(random_state=0,class_weight="balanced")
print(rf.fit(xtrain,ytrain))
ypred=(rf.predict(xtest))
ypred1=(rf.predict(xtrain))
print(ypred)
print(list(le.inverse_transform(ypred)))
print(classification_report(ypred,ytest))
print(rf.apply(xtest))
print(rf.decision_path(xtest))
print(rf.predict_proba(xtest))
print(rf.predict_log_proba(xtest))
print(rf.score(xtrain,ytrain))
rmse=math.sqrt(mean_squared_error(ytest,ypred))
print(rmse)
print(r2_score(ytest,ypred))
confusionmatrix=confusion_matrix(ypred,ytest)
print(confusionmatrix)
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
from sklearn.svm import SVC
svc=SVC(kernel="rbf",random_state=0,gamma=1,C=1,class_weight="balanced")
print(svc.fit(xtrain,ytrain))
ypred=svc.predict(xtest)
ypred1=svc.predict(xtrain)
print(ypred)
print(list(le.inverse_transform(ypred)))
print(svc.decision_function(xtest))
print(svc.score(xtrain,ytrain))
print(classification_report(ytest,ypred))
rmse=math.sqrt(mean_squared_error(ytest,ypred))
print(rmse)
print(r2_score(ytest,ypred))
confusionmatrix=confusion_matrix(ypred,ytest)
print(confusionmatrix)
##print(accuracy_score(ytest,ypred)
print(accuracy_score(ytrain,ypred1))
from sklearn.neural_network import MLPClassifier 
mlp=MLPClassifier(hidden_layer_sizes=(30,30))
print(mlp)
print(mlp.fit(xtrain,ytrain))
ypred=mlp.predict(xtest)
ypred1=mlp.predict(xtrain)
print(ypred)
print(list(le.inverse_transform(ypred)))
print(mlp.predict_proba(xtest))
print(mlp.predict_log_proba(xtest))
print(mlp.score(xtrain,ytrain))
print(classification_report(ytest,ypred))
rmse=math.sqrt(mean_squared_error(ytest,ypred))
print(rmse)
print(r2_score(ytest,ypred))
print(accuracy_score(ytest,ypred))
print(accuracy_score(ytrain,ypred1))
from keras.utils import to_categorical
import pandas as pd 
import numpy as np 
col_names=["Type","Posts"]
data=pd.read_csv("C:\\Users\\Saarthu\\Downloads\\project.csv",names=col_names,engine='python',sep=',',quotechar='"',error_bad_lines=False)
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from cleantext import clean
y=data.iloc[:,0]
x=data.iloc[:,1]
(clean(x,fix_unicode=True,lower=True,no_line_breaks=True,no_urls=True,no_emails=True,no_phone_numbers=True,no_numbers=True,no_digits=True,no_currency_symbols=True,no_punct=True))                                                                                                                    

from sklearn import preprocessing
le=preprocessing.LabelEncoder()
(le.fit(y))
y=(le.transform(y))
y=to_categorical(y)
print(y)
max_features=15000
max_words=50
batch_size=100
epochs=7
num_classes=4
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=50)
from keras .preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense,GRU,LSTM,Embedding
from keras.layers import SpatialDropout1D,Dropout,Bidirectional,Conv1D,GlobalMaxPooling1D,MaxPooling1D,Flatten
from keras.callbacks import ModelCheckpoint,TensorBoard,Callback,EarlyStopping
tokenizer=Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(xtrain))
xtrain=tokenizer.texts_to_sequences(xtrain)
#print(xtrain)
xtest=tokenizer.texts_to_sequences(xtest)
#print(xtest)
import pickle
with open("vectorizer_old","wb") as output:
 pickle.dump(tokenizer,output)
xtrain=pad_sequences(xtrain,maxlen=max_words)
xtest=pad_sequences(xtest,maxlen=max_words) 
print(xtrain,xtest)
from imblearn.over_sampling import SMOTE
sm=SMOTE(random_state=50)
xtrain,ytrain=sm.fit_resample(xtrain,ytrain)
print(xtrain,ytrain)
from keras.layers.core import Reshape
model_GRU=Sequential()
model_GRU.add(Embedding(max_features,100,mask_zero=True)) 
model_GRU.add(GRU(64,dropout=0.4,return_sequences=True))
model_GRU.add(GRU(32,dropout=0.5,return_sequences=False))
input_shape=y[0].shape
model_GRU.add(Dense(num_classes,activation="tanh",input_shape=input_shape))
from keras.optimizers import Adam
opt=Adam(lr=0.01) 
model_GRU.compile(loss='categorical_crossentropy',optimizer=opt,metrics=["categorical_accuracy"])
model_GRU.summary()	
history1=model_GRU.fit(xtrain,ytrain,epochs=epochs,batch_size=batch_size,verbose=1)
ypred=model_GRU.predict(xtest,verbose=1)
print(ypred)
ynum=np.argmax(ytest,axis=1)
from sklearn.metrics import accuracy_score,mean_squared_error,r2_score,classification_report,f1_score
#print(accuracy_score(ypred,ytest))
model3_LSTM=Sequential()
model3_LSTM.add(Embedding(max_features,100,mask_zero=True)) 
model3_LSTM.add(LSTM(64,dropout=0.4,return_sequences=True))
model3_LSTM.add(LSTM(32,dropout=0.5,return_sequences=False))
input_shape=y[0].shape
model3_LSTM.add(Dense(num_classes,activation="sigmoid",input_shape=input_shape))
from keras.optimizers import Adam
opt=Adam(lr=0.01) 
model3_LSTM.compile(loss='categorical_crossentropy',optimizer=opt,metrics=["categorical_accuracy"])
model3_LSTM.summary()	
history1=model3_LSTM.fit(xtrain,ytrain,epochs=epochs,batch_size=batch_size,verbose=1)
ypred=model_GRU.predict(xtest,verbose=1)
print(ypred)
ynum=np.argmax(ytest,axis=1)
#print(accuracy_score(ypred,ytest)) 
with open("lstm_model_old","wb") as output:
 pickle.dump(model3_LSTM,output)
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report

class PredictSentiment:
	def __init__(self):
		self.model = None
		self.vectorizer = None
		with open("lstm_model_old", 'rb') as data:
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
		x_test = pad_sequences(x_test, maxlen=50)


		
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
	print(pred.input("enter a post"))


