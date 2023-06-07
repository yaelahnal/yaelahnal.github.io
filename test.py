#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("dataset_kompas.csv")
from IPython.display import display


# In[ ]:





# In[26]:


df['judul'] = df['judul'].str.lower()
df['isi'] = df['isi'].str.lower()
df.head()


# In[27]:


import string 
import re #regex library
# import word_tokenize & FreqDist from NLTK

from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# ------ Tokenizing ---------

def remove_special(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\'," ").replace('\\f'," ").replace('\\r'," ")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    return text.replace("http://", " ").replace("https://", " ")
                
df['judul'] = df['judul'].apply(remove_special)
df['judul']

df['isi'] = df['isi'].apply(remove_special)




# In[28]:


#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

df['judul'] = df['judul'].apply(remove_number)
df['judul']
df['isi'] = df['isi'].apply(remove_number)


# In[29]:


#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

df['judul'] = df['judul'].apply(remove_punctuation)
df['judul']

df['isi'] = df['isi'].apply(remove_punctuation)


# In[30]:


#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

df['judul'] = df['judul'].apply(remove_whitespace_LT)
df['isi'] = df['isi'].apply(remove_whitespace_LT)


#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

df['judul'] = df['judul'].apply(remove_whitespace_multiple)
df['judul']

df['isi'] = df['isi'].apply(remove_whitespace_multiple)


# In[31]:


# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", " ", text)

df['judul'] = df['judul'].apply(remove_singl_char)
df['judul']

df['isi'] = df['isi'].apply(remove_singl_char)


# In[32]:


import nltk
nltk.download('punkt')
# NLTK word Tokenize 
def word_tokenize_wrapper(text):
    return word_tokenize(text)

df['judul'] = df['judul'].apply(word_tokenize_wrapper)
df['judul']

df['isi'] = df['isi'].apply(word_tokenize_wrapper)
df.to_csv("tempkompas.csv")


# In[33]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')


# In[34]:


list_stopwords = stopwords.words('indonesian')

#Menghapus Stopword dari list token
def stopwords_removal(words):
    return [word for word in words if word not in list_stopwords]

df['judul'] = df['judul'].apply(stopwords_removal)
df['isi'] = df['isi'].apply(stopwords_removal)
df.head()
df.to_csv("tempkompas.csv")


# In[35]:


from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import swifter


# create stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# stemmed
def stemmed_wrapper(term):
    return stemmer.stem(term)

term_dict = {}

for document in df['judul']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")

for document in df['isi']:
    for term in document:
        if term not in term_dict:
            term_dict[term] = ' '
            
print(len(term_dict))
print("------------------------")


# In[36]:


for term in term_dict:
    term_dict[term] = stemmed_wrapper(term)
    print(term,":" ,term_dict[term])
    
print(term_dict)
print("------------------------")


# apply stemmed term to dataframe
def get_stemmed_term(document):
    return [term_dict[term] for term in document]

df['judul'] = df['judul'].swifter.apply(get_stemmed_term)
df['isi'] = df['isi'].swifter.apply(get_stemmed_term)


# In[37]:


df.to_csv("hasil_processing.csv")


# In[38]:


dff = pd.read_csv("hasil_processing.csv")
dff.head()


# In[39]:


kategori_wisata = {
    'kuliner': ['restoran', 'kafe', 'warung', 'kuliner', 'seafood', 'steak', 'pizza', 'sushi', 'dimsum', 'bakery', 'kue', 'sate', 'burger', 'kebab', 'pasta', 'ramen', 'bubur', 'gado-gado', 'nasi', 'bakso'],
    'alam': ['gunung', 'pantai', 'hutan', 'air terjun', 'danau', 'gua', 'pegunungan', 'savana', 'desert', 'kawah', 'lembah', 'rawa', 'pulau', 'taman', 'gletser', 'bukit', 'sungai', 'laut', 'kepulauan', 'risiko'],
    'kota': ['tugu', 'pasar', 'monumen', 'museum', 'perbelanjaan', 'galeri', 'kafe', 'universitas', 'teater', 'stadion', 'olahraga', 'keramaian', 'bisnis', 'mal', 'jembatan', 'kolam renang', 'hiburan', 'anak', 'pelabuhan', 'transportasi'],
    'budaya': ['candi', 'keraton', 'seni', 'pasar', 'tari', 'tradisi', 'festival', 'musik', 'teater', 'kesenian', 'pahatan', 'lukisan', 'sastra', 'budaya', 'upacara', 'adat', 'pertunjukan', 'wayang', 'karnaval', 'patung'],
    'pendidikan': ['universitas', 'kampus', 'sekolah', 'perpustakaan', 'laboratorium', 'pelatihan', 'kursus', 'pendidikan', 'akademi', 'penelitian', 'studi'],
    'religius': ['kuil', 'gereja', 'masjid', 'pura', 'vihara', 'santuario', 'ibadah', 'makam', 'katedral', 'basilika', 'sinagoge', 'pagoda', 'monasteri', 'kloster', 'ziarah'],
    'hiburan': ["Taman", "Akuarium", "Hiburan", "Rekreasi", "Bermain", "Pertunjukan", "Teater", "Konser", "Festival", "Pameran", "Diskotek", "Klub", "Karaoke", "Bar", "Musik", "Renang"],
    'petualagan': ["Gunung", "Hutan", "Gua", "Laut", "Pantai", "Air", "Selam", "Panjat", "Rafting", "Trekking", "Ekspedisi", "Hiking", "Bukit", "Savana", "Jelajah", "Jungle", "Caving", "Terjun", "Jeram", "Ekplorasi"]
}


# In[40]:


kategori = []
for i in dff['isi'].tolist():
    temp = 'tidak terkategori'
    splitt = i.split(",")
    kuliner = 0
    alam = 0
    kota = 0
    budaya = 0
    pendidikan = 0
    religius = 0
    hiburan = 0 
    petualagan = 0
    for k in range (len(splitt)):
        splitt[k] = splitt[k].replace("'", "")
        splitt[k] = splitt[k].replace("[", "")
        splitt[k] = splitt[k].replace("]", "")
        splitt[k] = splitt[k].replace(" ", "")
    for j in splitt:
        if j in kategori_wisata['kuliner']:
            kuliner+=1
        elif j in kategori_wisata['alam']:
            alam+=1
        elif j in kategori_wisata['kota']:
            kota+=1
        elif j in kategori_wisata['budaya']:
            budaya+=1
        elif j in kategori_wisata['pendidikan']:
            pendidikan+=1
        elif j in kategori_wisata['religius']:
            religius+=1
        elif j in kategori_wisata['hiburan']:
            hiburan+=1
        elif j in kategori_wisata['petualagan']:
            petualagan+=1  
    print(f"kuliner: {kuliner}, alam: {alam}, kota: {kota}, budaya:{budaya}, pendidikan:{pendidikan}, religius:{religius}, hiburan:{hiburan}, petualagan: {petualagan} ")
    if kuliner > alam and kota > budaya > pendidikan > religius > hiburan > petualagan:
        temp = 'kuliner'
    elif alam > kuliner and alam > kota and alam > budaya and  alam > pendidikan and alam > religius and alam > hiburan and alam > petualagan  : 
        temp = 'alam'
    elif kota > kuliner and kota > alam  and kota > budaya and kota > pendidikan and kota > religius and kota > hiburan and kota > petualagan :
        temp = 'kota'
    elif budaya      > kuliner and budaya > alam and budaya > kota and budaya > pendidikan and budaya > religius and budaya > hiburan and budaya > petualagan:
        temp = "budaya"
    elif pendidikan  > kuliner and pendidikan > alam and pendidikan > kota and pendidikan > budaya and pendidikan >  religius and pendidikan > hiburan and pendidikan > petualagan :
        temp = "pendidikan"
    elif religius     > kuliner and religius > alam and religius > kota and religius > budaya and religius > pendidikan and religius > hiburan and religius > petualagan :
        temp = "religius"
    elif hiburan   > kuliner and hiburan > alam and hiburan > kota and budaya and hiburan > pendidikan and hiburan > religius and hiburan > petualagan :
        temp = "hiburan"
    elif petualagan     > kuliner and petualagan > alam and petualagan > kota and petualagan > budaya and petualagan > pendidikan and petualagan > religius and petualagan > hiburan :
        temp = "petualagan"
    kategori.append(temp)

dff['kategori'] = kategori

pd.set_option('display.max_rows', None)
dff


# In[41]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
dff


# In[42]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Bagi dataset menjadi data pelatihan dan data pengujian
X_train, X_test, y_train, y_test = train_test_split(dff['isi'], dff['kategori'], test_size=0.3, random_state=42)

# Ekstraksi fitur menggunakan TF-IDF
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Latih model KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_tfidf, y_train)

# Prediksi label untuk data uji
y_pred = knn.predict(X_test_tfidf)

# Evaluasi akurasi model
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi: {:.2f}%".format(accuracy * 100))


# In[43]:


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

# Langkah 5: Menampilkan hasil evaluasi
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# In[44]:




