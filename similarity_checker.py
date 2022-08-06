import re
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas
import numpy

#current structure:
#first value in the similarity array is the skillsforall course content for a specific course
#all other values are linkedin course contents

raw_data = []

#get all data from data.csv (or specific columns if wanted)
pdatabase_linkedin = pandas.read_csv("data.csv", usecols = ['Company', 'Job Title', 'Description'])
#pdatabase_linkedin = pandas.read_csv("data.csv", usecols = ['Description'])
pdatabase_skillsforall = pandas.read_csv("skillsforall_data.csv", usecols = ['Course Content'])

#append doesnt work for some reason
raw_data_linkedin = (pdatabase_linkedin.values.tolist())
raw_data_skillsforall = (pdatabase_skillsforall.values.tolist())

#right now only one index in skillsforall, compare all linkedin (values after (1,1)) to it (see note at top of file)
for data in raw_data_skillsforall:
    raw_data.append(data)
for data in raw_data_linkedin:
    raw_data.append([data[2]]) #select 'Description', put in [] to make it work with for in loop in nlp_clean (i think?)

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
    return(data)
#end nlp_clean

data = nlp_clean(raw_data)


#NOTE: see note at top of file
def cos_similarity(data):
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    #https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
    vect = TfidfVectorizer(min_df=1, stop_words="english", strip_accents="ascii")                                                                                                                                                                                                   
    tfidf = vect.fit_transform(data)                                                                                                                                                                                                                       
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity
#end nlp_similarity


#run to get best match between first value (skillsforall) and all linkedin jobs
#NOTE: in general case, this finds the closest index to the first value in the comparison array, raw_data
#NOTE: in current case, this finds the closest index to a skillsforall course in the first slot
def best_match(similarity_matrix):
    #select first row of similarity matrix & pop first elem out b/c it's 1
    first_row = similarity_matrix[0][1:]
    closest_index = find_nearest(first_row,1)
    closest_company = raw_data_linkedin[closest_index][0]
    closest_job_title = raw_data_linkedin[closest_index][1]
    return [closest_company, closest_job_title]

#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
#this is generalized, pass in 1 most of the time, see note in best_match
def find_nearest(array, value):
    array = numpy.asarray(array)
    idx = (numpy.abs(array - value)).argmin()
    return idx #array[idx]


similarity = cos_similarity(data).toarray() #this variable is inefficient, translate sparse array to numpy array

#print(similarity)
print("----------")
best_job = best_match(similarity)
print("The closest job to this course is from " + str(best_job[0]) + ", as a " + str(best_job[1]) + ".")

#print("end program")