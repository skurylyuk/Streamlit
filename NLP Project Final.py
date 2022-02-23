#!/usr/bin/env python
# coding: utf-8

# In[974]:


# import important modules
import numpy as np
import pandas as pd

# sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB # classifier 

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    plot_confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# text preprocessing modules
from string import punctuation 

# text preprocessing modules
from nltk.tokenize import word_tokenize

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
import re #regular expression

import warnings
warnings.filterwarnings("ignore")
# seeding
np.random.seed(123)
from sklearn.decomposition import NMF


# In[975]:


df = pd.read_csv('SPAM text message 20170820 - Data.csv', encoding="latin-1")


# In[976]:


df


# In[978]:


df.Category.value_counts()


# In[979]:


df = df.drop_duplicates()
df = df.reset_index(inplace = False)[['Category','Message']]
print(df.shape)


# In[980]:


df['labels'] = df['Category'].map({'ham':0, 'spam':1})
df


# In[981]:


spam_messages= df.loc[df.Category=="spam"]["Message"]
not_spam_messages= df.loc[df.Category=="ham"]["Message"]

print("spam count: " +str(len(df.loc[df.Category=="spam"])))
print("not spam count: " +str(len(df.loc[df.Category=="ham"])))

not_spam_messages


# In[1053]:


spam_messages


# In[1067]:


spam_messages_df = pd.DataFrame(spam_messages)
spam_messages_df


# In[1073]:


spam_messages_df['character_Count'] = df['Message'].str.len()
spam_messages_df.sort_values(by='character_Count', inplace = True)


# In[1095]:


spam_messages_df.head(50)


# In[1097]:


spam_messages_df[spam_messages_df["Message"].str.contains('free')].head(20)


# In[1100]:


df.loc[3247
      ].Message


# In[1060]:


df.loc[93].Message
#df.loc[8].Clean_Message_2


# # Text Preprocessing 

# In[984]:


def preprocessing_text(texts):
    df["Clean_Message"] = df["Message"].str.lower() #puts everything in lowercase
    df["Clean_Message"] = df["Message"].replace(r'http\S+', '', regex=True) # removing any links 
    df["Clean_Message"] = df["Message"].replace(r'www.[^ ]+', '', regex=True)
    df["Clean_Message"] = df["Message"].replace(r'[0-9]+', " ", regex = True) #removing numbers
    df["Clean_Message"] = df["Message"].replace (r'[!"#$%&()*+,-./:;<=>?@[\]^_`{|}~]', '', regex=True) #remove special characters and puntuation marks 
    df["Clean_Message"] = df["Message"].replace(r"[^A-Za-z]", " ", regex = True) #replace any item that is not a letter
    

    return texts


# In[985]:


texts = preprocessing_text(df)
texts


# In[987]:


stop_words =  stopwords.words('english')

def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # Clean the text, with the option to remove stop_words and to lemmatize word

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9]", " ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) # remove numbers
    text = text.lower()
    
        
    #Remove punctuation from text
    text = ''.join([c for c in text if c not in punctuation])
    
    # Optionally, remove stop words
    if remove_stop_words:
        text = text.split()
        text = [w for w in text if not w in stop_words]
        text = " ".join(text)
    
    # Optionally, shorten words to their stems
    if lemmatize_words:
        text = text.split()
        lemmatizer = WordNetLemmatizer() 
        lemmatized_words = [lemmatizer.lemmatize(word) for word in text]
        text = " ".join(lemmatized_words)
    
    # Return a list of words
    return(text)


# In[988]:


#clean the review
df["Clean_Message_2"]= df["Message"].apply(text_cleaning)


# In[990]:


spam_list = df[df["Category"] == "spam"]["Message"].unique().tolist()
spam = " ".join(spam_list)
spam_wordcloud = WordCloud().generate(spam)
plt.figure(figsize=(12,8))
plt.imshow(spam_wordcloud)
plt.show()


# In[991]:


ham_list = df[df["Category"] == "ham"]["Message"].unique().tolist()
ham = " ".join(ham_list)
ham_wordcloud = WordCloud().generate(ham)
plt.figure(figsize=(12,8))
plt.imshow(ham_wordcloud)
plt.show()


# # Modeling - TFIDF Vectorization
# 

# In[996]:


#TFIDF 

docs = df.Clean_Message_2
tfidf= TfidfVectorizer(stop_words= "english",
                       max_df = .4, 
                       min_df = 5, #maybe 5 or 6
                       max_features = 20000,
                       lowercase=True, 
                       analyzer='word',
                       ngram_range=(1,3),
                       dtype=np.float32)
doc_term_matrix = tfidf.fit_transform(docs) #should this be values?

#


# In[997]:


doc_term_matrix = pd.DataFrame(doc_term_matrix.toarray(), columns = tfidf.get_feature_names())
doc_term_matrix.head()


# In[998]:


doc_term_matrix.shape


# In[999]:


doc_term_matrix


# In[1000]:


docs


# In[1001]:


vocabulary = tfidf.get_feature_names()
vocabulary


# # Topic Modeling - NMF

# In[1002]:


nmf = NMF(n_components = 10)
topic_doc= nmf.fit_transform(doc_term_matrix)
topic_doc.shape 


# In[1003]:


topic_doc[0] # topic representation for first text classifier fit topic doc y - classifier.fit(topic_doc, y)


# In[1004]:


topic_matrix = nmf.components_
topic_matrix.shape


# In[1005]:


word_topic_matrix_df = pd.DataFrame(nmf.components_ , columns = vocabulary).T.add_prefix('topic_')
word_topic_matrix_df


# In[1015]:


word_topic_matrix_df.sort_values(by = ['topic_2'], ascending=False).head(5)


# # Topic Modeling Version 1

# In[1101]:


def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        
display_topics(nmf, tfidf.get_feature_names(), 10)


# # Topic Modeling Verison 2

# In[1025]:


topics_df= pd.DataFrame([display_topics(nmf, tfidf.get_feature_names(), 10)])


# In[1028]:


sub ='like it gt'
 
# start var
start = 2
 
# creating and passing series to new column
df["Clean_Message_2"].str.find(sub, start)
 
df[df["Clean_Message_2"].str.contains('like it gt')]


# In[1046]:


df[df["Clean_Message_2"].str.contains('gt')].tail(20)


# In[1102]:


df.loc[55].Message
df.loc[55].Clean_Message_2


# # Topic Modeling Verison 3

# In[646]:


docs_2 = df.Clean_Message
tfidf2= TfidfVectorizer(stop_words= "english",
                       max_df = .6, 
                       min_df = 5, #maybe 5 or 6
                       max_features = 20000,
                       lowercase=True, 
                       analyzer='word',
                       ngram_range=(1,3),
                       dtype=np.float32)
doc_term_matrix2 = tfidf2.fit_transform(docs_2) #should this be values?


# In[1103]:


display_topics(nmf, tfidf2.get_feature_names(), 10)


# # Topic Modeling Version 4

# In[648]:


docs_2 = df.Clean_Message
tfidf2= TfidfVectorizer(stop_words= "english",
                        max_df = .75, 
                       min_df = 5, #maybe 5 or 6
                       max_features = 2000,
                       lowercase=True, 
                       analyzer="word", #changed
                       ngram_range=(1,3))
doc_term_matrix2 = tfidf2.fit_transform(docs_2) #should this be values?


# In[1104]:


display_topics(nmf, tfidf2.get_feature_names(), 10)


# # Split Data in to Test Train

# In[654]:


X= df['Clean_Message_2']
y= df['labels']


# In[656]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.15, random_state=42, shuffle = True, stratify = y)


# In[657]:


X_train_vect = tfidf.fit_transform(X_train)
X_test_vect = tfidf.transform(X_test)


# In[658]:


X_train_vect.shape


# In[659]:


X_test_vect.shape


# In[660]:


print(X_train_vect.toarray())


# # Classification Algorithmns

# In[661]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=1000, criterion = "entropy", random_state = 0)
rf.fit(X_train_vect, y_train)


# In[662]:


y_pred = rf.predict(X_test_vect)


# In[663]:


from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
print("\nConfusion Matrix\n",confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[688]:


features_train, features_test, labels_train, labels_test = train_test_split(features, df['Category'], test_size=0.3, random_state=111)


# In[689]:


from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


# In[674]:


svc = SVC(kernel='sigmoid', gamma=1.0)
knc = KNeighborsClassifier(n_neighbors=49)
mnb = MultinomialNB(alpha=0.2)
dtc = DecisionTreeClassifier(min_samples_split=7, random_state=111)
lrc = LogisticRegression(solver='liblinear', penalty='l1')
rfc = RandomForestClassifier(n_estimators=31, random_state=111)
abc = AdaBoostClassifier(n_estimators=62, random_state=111)
bc = BaggingClassifier(n_estimators=9, random_state=111)
etc = ExtraTreesClassifier(n_estimators=9, random_state=111)


# In[765]:


clfs = {'SVC' : svc,'KNeighbors' : knc, 'MultinomialNB': mnb, 'DecisionTree': dtc, 'Logistic Regression': lrc, 'Random Forest': rfc, 'AdaBoost': abc, 'Bagging': bc, 'Extra Trees': etc}


# In[766]:


def train_classifier(clf, feature_train, labels_train):    
    clf.fit(feature_train, labels_train)


# In[768]:


features = tfidf.fit_transform(df['Message'])


# In[769]:


def predict_labels(clf, features):
    return (clf.predict(features))


# In[770]:


pred_scores = []
for k,v in clfs.items():
    train_classifier(v, features_train, labels_train)
    pred = predict_labels(v,features_test)
    pred_scores.append((k, accuracy_score(labels_test,pred)))
pred_scores


# In[771]:


df_class = pd.DataFrame(pred_scores, columns=["Classifier", "Accuracy_Score"])
df_class = df_class.sort_values(by="Accuracy_Score", ascending= True)
df_class


# In[807]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axis import Axis
import seaborn as sns

# create dataset
height = df_class["Accuracy_Score"]
bars = df_class["Classifier"]
x_pos = np.arange(len(height))
 
fig,ax = plt.subplots(figsize=(20,15))
sns.barplot(x_pos, height, ci=95,ax=ax, palette = "plasma")


#plt.xlim(8, 8)
plt.ylim(.85, 1.00)

plt.title('Classifier Comparison', fontsize=35)
#plt.xlabel('Classifier', fontsize=14)
plt.ylabel('Accuracy Percentage', fontsize=14)
 
# Create names on the x axis
plt.xticks(x_pos, bars)
 
# Show graph


# In[809]:


svc = SVC(kernel='sigmoid', gamma=1.0)
svc.fit(X_train_vect, y_train)


# In[811]:


y_pred = svc.predict(X_test_vect)


# In[813]:


from sklearn.metrics import log_loss,confusion_matrix,classification_report,roc_curve,auc
print("\nConfusion Matrix\n",confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))


# In[851]:


import seaborn as sns
cf_matrix = confusion_matrix(y_test,y_pred)
sns.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# In[860]:


from sklearn.model_selection import RandomizedSearchCV

#Number of Trees in random forest
n_estimators= [int(x) for x in np.linspace(start =200, stop = 2000, num = 10)]

#Number of features to consider at every split 
max_features = ["auto", "sqrt"]

#Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10,110, num=11)]
max_depth.append(None)

#minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

#minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

#Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

pprint(random_grid)


# # Random Forest Hyperparamter Tuning 

# In[865]:



rf_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid,
                              n_iter = 10,
                              cv=3,
                              verbose =2,
                              random_state =42,
                              n_jobs = -1)

#Fit the random search model
rf_random.fit(features_train, labels_train)


# In[919]:


rf_random.best_params_


# In[873]:


rf=  RandomForestClassifier(n_estimators=1000, 
                                               min_samples_split= 10,
                                               min_samples_leaf=  1,
                                               max_features= "auto",
                                               max_depth = 80,
                                               bootstrap= False)
rf.fit(X_train_vect, y_train)
y_pred_rfc = rf.predict(X_test_vect)


# In[875]:


print("confusion matrix: \n\n", 
      confusion_matrix(y_test, y_pred_rfc))

print(classification_report(y_test, y_pred_rfc))


# # MultinomialNB Hyperparameter Tuning 

# In[942]:


mnb.get_params()


# In[951]:


mnb_param_grid = {'alpha': [0.01, 1.0, 10.0, 50.0]}
mnb_grid = GridSearchCV(MultinomialNB(), param_grid=mnb_param_grid, scoring = 'neg_log_loss')


# In[952]:


mnb_grid.fit(features_train, labels_train)


# In[928]:


mnb.get_params().keys()


# In[955]:


mnb_grid.best_params_


# In[958]:


mnb = MultinomialNB(alpha=0.01)
mnb.fit(features_train, labels_train)


# In[960]:


y_pred_mnb = mnb.predict(features_test)


# In[961]:


print("confusion matrix: \n\n", 
      confusion_matrix(labels_test, y_pred_mnb))

print(classification_report(labels_test, y_pred_mnb))


# # SVC Hyperparameter Tuning 
# 

# In[892]:


from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.model_selection import learning_curve,GridSearchCV


# In[895]:


param_grid = {'C':[1,10,100,1000],'gamma':[1,0.1,0.001,0.0001], 'kernel':['linear','rbf']}

SVC_grid = GridSearchCV(SVC(),param_grid,refit = True, verbose=2)


# In[897]:


SVC_grid.fit(features_train, labels_train)


# In[900]:


SVC_grid.best_params_


# In[903]:


svc = SVC(C= 10, kernel='rbf', gamma=1)
svc.fit(features_train, labels_train)


# In[911]:


y_pred_svc = svc.predict(features_test)


# In[964]:


print("confusion matrix: \n\n", 
      confusion_matrix(labels_test, y_pred_svc))

print(classification_report(labels_test, y_pred_svc))


# # Scatter Text

# In[827]:


import scattertext as st
import spacy 
from pprint import pprint


# In[829]:


scattertext_df =df[["Category", "Clean_Message_2"]]
scattertext_df


# In[832]:


#Scatter Text Corpus

nlp = spacy.load('en')
corpus = st.CorpusFromPandas(scattertext_df,
                            category_col ="Category",
                            text_col ="Clean_Message_2",
                            nlp=nlp).build()


# In[833]:


print(list(corpus.get_scaled_f_scores_vs_background().index[:10]))


# In[835]:


term_freq_df = corpus.get_term_freq_df()
term_freq_df['Spam Score'] = corpus.get_scaled_f_scores('spam')
pprint(list(term_freq_df.sort_values(by='Spam Score', ascending=False).index[:10]))


# In[840]:


term_freq_df['Normal Text Score'] = corpus.get_scaled_f_scores("ham")
pprint(list(term_freq_df.sort_values(by='Normal Text Score', ascending=False).index[:10]))


# In[850]:


html = st.produce_scattertext_explorer(corpus,
                                       category='spam',
                                       category_name='spam',
                                       #not_category_name='ham',
                                       width_in_pixels=1000)
                                       #metadata=scattertext_df['Clean_Message_2'])
open("Convention-Visualization.html", 'wb').write(html.encode('utf-8'))


# # Streamlit
# 

# In[971]:


from sklearn.neural_network import MLPClassifier
neural_net_pipeline = Pipeline([('vectorizer', tfidf), 
                                ('nn', MLPClassifier(hidden_layer_sizes=(700, 700)))])


# In[972]:


neural_net_pipeline.fit(X_train, y_train)


# In[970]:


from joblib import dump
dump(neural_net_pipeline, 'spam_classifier.joblib')

