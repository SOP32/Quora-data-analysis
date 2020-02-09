
# coding: utf-8

# In[ ]:


nltk.download()


# In[ ]:


nltk.download('punkt')


# In[1]:


import os
import numpy as np
import pandas as pd
import nltk
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
import string


# In[2]:


print(os.listdir("C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files"))


# In[3]:


print(os.listdir("C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/embeddings"))


# In[34]:


train = pd.read_csv('C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/train.csv')
test = pd.read_csv('C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/test.csv')


# In[5]:


train.head()


# In[6]:


test.head()


# In[7]:


train.shape


# In[8]:


test.shape


# In[11]:


train.describe()


# In[14]:


from nltk.tokenize import word_tokenize
word=[]
stopword = set(nltk.corpus.stopwords.words('english')) 
def clean(questions):
    for i in questions:
        #split into words
        i = word_tokenize(i)
        # convert to lower case
        i = [x.lower() for x in i]
        # remove punctuation from each word  
        punc = str.maketrans('', '', string.punctuation)
        removed = [x.translate(punc) for x in i]
        # remove remaining tokens that are not alphabetic
        words = [word for word in removed if word.isalpha()]
        # filter out stop words
        words = [x for x in words if not x in stopword]
        wordsall = ' '.join(words)
        word.append(wordsall)
    return pd.Series(word)


# In[15]:


insincere_questions = [text for text in train[train['target'] == 1]['question_text']]
print(insincere_questions[0])
insincere_questions_processed = clean(insincere_questions)
insincere_questions_processed = ' '.join(insincere_questions_processed).split()
#print(insincere_questions_processed)


# In[21]:


# Count all unique words
insincere_wordcount = Counter(insincere_questions_processed)
# get words and word counts
insincere_common_words = [word[0] for word in insincere_wordcount.most_common(15)]
insincere_common_word_count = [word[1] for word in insincere_wordcount.most_common(15)]
print(insincere_common_words)
# plot 20 most common words in insincere questions
sns.barplot(insincere_common_words, insincere_common_word_count, palette='husl')
#sns.despine(left=True, bottom=True)
sns.set(rc={'figure.figsize':(70,15)})
#plt.xlabel('')
#plt.ylabel('')
plt.title('Insincere Common Words', fontsize=55)
plt.tick_params(axis='x', which='major', labelsize=55)
plt.show()


# In[17]:


sincere_questions = [text for text in train[train['target'] == 0]['question_text']]
print(sincere_questions[0])
sincere_questions_processed = clean(sincere_questions)
sincere_questions_processed = ' '.join(sincere_questions_processed).split()
#print(sincere_questions_processed)


# In[22]:


# Count all unique words
sincere_wordcount = Counter(sincere_questions_processed)
# get words and word counts
sincere_common_words = [word[0] for word in sincere_wordcount.most_common(15)]
sincere_common_word_count = [word[1] for word in sincere_wordcount.most_common(15)]
print(sincere_common_words)
# plot 20 most common words in insincere questions
sns.barplot(sincere_common_words, sincere_common_word_count, palette='husl')
#sns.despine(left=True, bottom=True)
sns.set(rc={'figure.figsize':(70,15)})
#plt.xlabel('')
#plt.ylabel('')
plt.title('Sincere Common Words', fontsize=55)
plt.tick_params(axis='x', which='major', labelsize=55)
plt.show()


# In[ ]:


#i = [x.lower() for x in sentences]
#punc = str.maketrans('', '', string.punctuation)
#removed = [x.translate(punc) for x in i]
#words = [word for word in removed if word.isalpha()]
#words = [x for x in words if not x in stopword]


# In[39]:


train = pd.read_csv('C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/train.csv')
test = pd.read_csv('C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/test.csv')


# In[42]:


from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

sentences = np.array(train['question_text'])
#sentences2 = np.array(test['question_text'])
#sentences = np.concatenate((sentences1, sentences2), axis=0)
i = [x.lower() for x in sentences]
punc = str.maketrans('', '', string.punctuation)
removed = [x.translate(punc) for x in i]
sentences = [x for x in removed if not x in stopword]
vectorizer = CountVectorizer()
vectorizer.fit(sentences)
#for i in words:
#    train['clean'] = words[i]
#x = vectorizer.transform(train['clean'])
x = vectorizer.transform(train['question_text'])
y = train['target']
#X_test = vectorizer.transform(test['question_text'])
X_train, X_test, Y_train, Y_test= train_test_split(x, y, random_state= 0)

classifier1 = naive_bayes.MultinomialNB()
classifier2 = naive_bayes.BernoulliNB()
#fit_model= classifier.fit(X_train, train['target'])
#score = fit_model.score(X_test, test['target'])
fit_model1= classifier1.fit(X_train, Y_train)
score1 = fit_model1.score(X_test, Y_test)

fit_model2= classifier2.fit(X_train, Y_train)
score2 = fit_model2.score(X_test, Y_test)

print(score1)
print(score2)

prediction1 = fit_model1.predict(X_test)
print(f1_score(prediction1, Y_test, average= 'binary'))

prediction2 = fit_model2.predict(X_test)
print(f1_score(prediction2, Y_test, average= 'binary'))


# In[44]:


train = pd.read_csv('C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/train.csv')
test = pd.read_csv('C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/test.csv')


# In[45]:


from sklearn import naive_bayes
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

sentences = np.array(train['question_text'])
#sentences2 = np.array(test['question_text'])
#sentences = np.concatenate((sentences1, sentences2), axis=0)

vectorizer = CountVectorizer()
vectorizer.fit(sentences)
#for i in words:
#    train['clean'] = words[i]
#x = vectorizer.transform(train['clean'])
x = vectorizer.transform(train['question_text'])
y = train['target']
#X_test = vectorizer.transform(test['question_text'])
X_train, X_test, Y_train, Y_test= train_test_split(x, y, random_state= 0)

classifier1 = naive_bayes.MultinomialNB()
classifier2 = naive_bayes.BernoulliNB()
#fit_model= classifier.fit(X_train, train['target'])
#score = fit_model.score(X_test, test['target'])
fit_model1= classifier1.fit(X_train, Y_train)
score1 = fit_model1.score(X_test, Y_test)

fit_model2= classifier2.fit(X_train, Y_train)
score2 = fit_model2.score(X_test, Y_test)

print(score1)
print(score2)

prediction1 = fit_model1.predict(X_test)
print(f1_score(prediction1, Y_test, average= 'binary'))

prediction2 = fit_model2.predict(X_test)
print(f1_score(prediction2, Y_test, average= 'binary'))

