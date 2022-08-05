import re
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas
import numpy

#with open('data.csv', newline='') as csvfile:
#    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
#    for row in spamreader:
#        print(' '.join(row))

pdatabase = pandas.read_csv("data.csv", usecols = ['Description'])
#print(df)

raw_data = pdatabase.values.tolist()
#print(data)

#clean and prepare data for NLP
def nlp_clean(raw_data):
    data = []
    #https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/
    for desc in raw_data:
        try:
            desc = desc[0].lower()
            desc = re.sub(r'[^a-zA-Z0-9\s]', ' ', desc)
            desc = re.sub(r'\s{2,}', ' ', desc.strip())
            #inefficient, drop old list for mem or replace values
            data.append(desc)
        except:
            print("NaN")

    #print(data)
    return(data)
#end nlp_clean

data = nlp_clean(raw_data)

def cos_similarity(data):
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    #https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
    vect = TfidfVectorizer(min_df=1, stop_words="english", strip_accents="ascii")                                                                                                                                                                                                   
    tfidf = vect.fit_transform(data)                                                                                                                                                                                                                       
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity
#end nlp_similarity

similarity = cos_similarity(data).toarray() #this variable is inefficient, translate sparse array to numpy array

print(similarity)
print("end program")