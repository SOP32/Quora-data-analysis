
# coding: utf-8

# In[20]:


from sklearn import naive_bayes
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import naive_bayes
import pandas as pd
from collections import Counter

import numpy as np
from pandas import read_csv
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import matplotlib.pyplot as plt



df =pd.read_csv('C:/Users/Sohini Palit/Desktop/UPitt/Spring 2019/Machine Learning/Project/Project_files/train.csv')

#sentences = np.array(train['question_text'])
#vectorizer = CountVectorizer()
#vectorizer.fit(sentences)


x = df['question_text']
y = df.target

print("Number of questions: ", df.shape[0])
print(df.target.value_counts())
print("Percentage of insincere questions: {}".format(sum(df.target == 1)*100/len(df.target))) 


stopwords = set(STOPWORDS)
more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}
stopwords = stopwords.union(more_stopwords)
sincere_words = df[df.target==0].question_text.apply(lambda x: x.lower().split()).tolist()
insincere_words = df[df.target==1].question_text.apply(lambda x: x.lower().split()).tolist()

sincere_words = [item for sublist in sincere_words for item in sublist if item not in stopwords]
insincere_words = [item for sublist in insincere_words  for item in sublist if item not in stopwords ]

sincere_word = [sincere_word for sincere_word in sincere_words if sincere_word.isalpha()]
insincere_word = [insincere_word for insincere_word in insincere_words if insincere_word.isalpha()]
      
print('Number of sincere words',len(sincere_words))
print('Number of insincere words',len(insincere_words))



sincere_ngrams = zip(*[sincere_words[i:] for i in range(3)])
sincere_ngram_counter = Counter([" ".join(sincere_ngrams) for sincere_ngrams in sincere_ngrams])

insincere_ngrams = zip(*[insincere_words[i:] for i in range(3)])
insincere_ngram_counter = Counter([" ".join(insincere_ngrams) for insincere_ngrams in insincere_ngrams])

most_common_sincere_ngram = sincere_ngram_counter.most_common()[:9]
most_common_sincere_ngram = pd.DataFrame(most_common_sincere_ngram)
most_common_sincere_ngram.columns = ['word', 'freq']
most_common_sincere_ngram['percentage'] = most_common_sincere_ngram.freq *100 / sum(most_common_sincere_ngram.freq)
print(most_common_sincere_ngram)


most_common_insincere_ngram = insincere_ngram_counter.most_common()[:9]
most_common_insincere_ngram = pd.DataFrame(most_common_insincere_ngram)
most_common_insincere_ngram.columns = ['word', 'freq']
most_common_insincere_ngram['percentage'] = most_common_insincere_ngram.freq *100 / sum(most_common_insincere_ngram.freq)
print(most_common_insincere_ngram)


# In[21]:


sns.barplot(most_common_sincere_ngram.word, most_common_sincere_ngram.freq, palette='husl')
#sns.despine(left=True, bottom=True)
sns.set(rc={'figure.figsize':(70,15)})
#plt.xlabel('')
#plt.ylabel('')
plt.title('Sincere Common Words', fontsize=55)
plt.tick_params(axis='x', which='major', labelsize=55, rotation=90)
plt.show()


# In[23]:


sns.barplot(most_common_insincere_ngram.word, most_common_insincere_ngram.freq, palette='husl')
#sns.despine(left=True, bottom=True)
sns.set(rc={'figure.figsize':(70,15)})
#plt.xlabel('')
#plt.ylabel('')
plt.title('Insincere Common Words', fontsize=55)
plt.tick_params(axis='x', which='major', labelsize=55, rotation=90)
plt.show()

