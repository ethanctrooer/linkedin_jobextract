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
        except Exception as ex:
            #print("NaN")
            template = "An exception of type {0} occurred. Arguments:\n{1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print(message)
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
def best_match(similarity_matrix, num_matches):
    #select first row of similarity matrix & pop first elem out b/c it's 1
    retval = []
    first_row = similarity_matrix[0][1:]

    if len(first_row) < num_matches:
        print("ERROR: number of requested matches exceeds avaliable data")
        return

    closest_indexes = find_nearest(first_row,1,num_matches) #change last value to change number of jobs grabbed
    for index in closest_indexes:
        retval.append([raw_data_linkedin[index][0],raw_data_linkedin[index][1]]) #append company name, job title
    return retval

#https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
#this is generalized, pass in 1 most of the time, see note in best_match
#array: pass in similarity array, value: value to be closest to, howmany: number of indexes returned
def find_nearest(array, value, howmany):
    if len(array) < howmany:
        print("ERROR: number of requested nearest exceeds avaliable data")
        return
    array = numpy.asarray(array)
    indexes = []
    for i in range(howmany):
        idx = (numpy.abs(array - value)).argmin()
        indexes.append(idx)
        array = numpy.delete(array, idx)
    return indexes #array[idx]


similarity = cos_similarity(data).toarray() #this variable is inefficient, translate sparse array to numpy array

#print(similarity)
print("----------")

best_jobs = best_match(similarity, 3) #second number for how many closest jobs to grab

for idx, job in enumerate(best_jobs): #use enumerate to get index
    #add 1 to index to convert from programmer leetcode to normal person
    print("The #" + str(idx+1) + " closest job to this course is from " + str(job[0]) + ", as a " + str(job[1]) + ".")

#print("end program")

#TODO: get job links from linkedin & display alongside these
#TODO: output to file