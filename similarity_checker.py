import re
from tokenize import String
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import pandas
import numpy
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction._stop_words import ENGLISH_STOP_WORDS
from scipy import spatial
import gensim.downloader as api
# https://gist.github.com/ethen8181/d57e762f81aa643744c2ffba5688d33a

#current structure:
#first value in the similarity array is the skillsforall course content for a specific course
#all other values are linkedin course contents

raw_data = []

#get all data from data.csv (or specific columns if wanted)
pdatabase_linkedin = pandas.read_csv("data_v3_pair_goodish_sample.csv", usecols = ['Company', 'Job Title', 'Description'])
#pdatabase_linkedin = pandas.read_csv("data.csv", usecols = ['Description'])
pdatabase_skillsforall = pandas.read_csv("skillsforall_data_v3_pair_goodish_sample.csv", usecols = ['Course Content'])

#append doesnt work for some reason
raw_data_linkedin = (pdatabase_linkedin.values.tolist())
raw_data_skillsforall = (pdatabase_skillsforall.values.tolist()) 

temp_skillsforall_data = []
#combine all values into one value for processing
for elem in raw_data_skillsforall:
    temp_skillsforall_data.append(str(elem))
#end for loop
str_data_skillsforall = ''.join(temp_skillsforall_data) #this might not be the best solution

#right now only one index in skillsforall, compare all linkedin (values after (1,1)) to it (see note at top of file)
raw_data.append(str_data_skillsforall)
#for data in temp_skillsforall_data: 
#    raw_data.append(data)
for data in raw_data_linkedin:
    raw_data.append([data[2]]) #select 'Description', put in [] to make it work with for in loop in nlp_clean (i think?)

#clean and prepare data for NLP
#only works with LISTS/ARRAYS of strings, not strings themselves
def nlp_clean(raw_data):
    data = []
    #https://albertauyeung.github.io/2018/06/03/generating-ngrams.html/
    for desc in raw_data:
        try:
            #print(type(desc))
            if type(desc) == str: #do this to accomidate the skillsforall string
                desc = re.sub(r'[^a-zA-Z0-9\s]', ' ', desc)
                desc = re.sub(r'\s{2,}', ' ', desc.strip())
                desc = desc.lower()
                desc = desc.replace(u'xa0', u'') #remove weird whitespace artifact from beautifulsoup
                #inefficient, drop old list for mem or replace values
                data.append(desc)
            else:
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

#this is mostly for analysis used in conjunction with get_word_freq, doesnt affect actual program
def word_preprocess(input_string):
    text_tokens = word_tokenize(input_string)
    STOP_WORDS = set( stopwords.words('english') ).union( set(ENGLISH_STOP_WORDS) )

    tokens_without_sw = [word for word in text_tokens if not word in STOP_WORDS]

    #print(tokens_without_sw)
    return tokens_without_sw
#end word_preprocess


#pass in string
def get_word_freq(input_string):
    string_data = input_string.split()
    #for sentence in data_in_one:
    word_occurances = dict()
    while len(string_data) > 0:
        #string_data[0] is first word in the list at the time
        num_occurances = int(string_data.count(string_data[0])) #not sure if int is nessecary
        word_occurances[string_data[0]] = num_occurances

        #remove all instances of the list
        # filter the list to exclude 0's
        string_data = [i for i in string_data if i != string_data[0]]
    #end while loop
    k = (dict(reversed(sorted(word_occurances.items(), key=lambda item: item[1]))))
    #print(k)
    return k
#end get_word_freq

#------------------------------begin similarity functions------------------------------

#NOTE: see note at top of file
def cos_similarity(data):
    #https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
    #https://stackoverflow.com/questions/8897593/how-to-compute-the-similarity-between-two-text-documents
    STOP_WORDS = set( stopwords.words('english') ).union( set(ENGLISH_STOP_WORDS) )
    #^^not great, uses default set

    vect = TfidfVectorizer(min_df=2, stop_words=STOP_WORDS, strip_accents="ascii") #can add argument stop_words="english" here, but not what we need (auto removes some wanted words, like target)                                                                                                                                                                                              
    tfidf = vect.fit_transform(data)
    pairwise_similarity = tfidf * tfidf.T
    return pairwise_similarity
#end cos_similarity


#pass in array, compare 1st index to all others and return each value's similarity in same order
#https://newscatcherapi.com/blog/ultimate-guide-to-text-similarity-with-python
def jaccard_similarity(data):
    comparison_point = data[0]
    comparison_values = []
    for compare in data:
        intersection_cardinality = len(set.intersection(*[set(comparison_point), set(compare)]))
        union_cardinality = len(set.union(*[set(comparison_point), set(compare)]))
        comparison_values.append(intersection_cardinality/float(union_cardinality))
    return comparison_values
#end jaccard_similarity

# 4 in https://stackoverflow.com/questions/65852710/text-similarity-using-word2vec
#dataset omits words like blueorigin, glassdoor, etc. is an old dataset and not customized, has limitations
#probably bugged but whatever
def word2vec_similarity(data):
    #get datasets from:
    #https://github.com/RaRe-Technologies/gensim-data/releases?page=1
    #https://github.com/alexandres/lexvec#pre-trained-vectors
    #or train your own
    #model = api.load("fasttext-wiki-news-subwords-300")
    model = api.load("glove-wiki-gigaword-50")
    retval = []

    #put all word vectors in array then make each value in input into vector
    str_as_array = []
    for value in data:
        print("")
        x = value.split()
        word_vecs = [] #getting word vectors from model
        for i in x:
            try:
                word_vecs.append(model[i])
            except Exception as ex:
                template = "An exception of type {0} occurred. Arguments:\n{1!r}"
                message = template.format(type(ex).__name__, ex.args)
                print(message)
        #end for loop
        str_as_array.append(numpy.sum(numpy.array(word_vecs), axis=0))
        
    #compare vectors to starting vector then return
    for value in str_as_array:
        close_score = 1 - spatial.distance.cosine(str_as_array[0], value)
        retval.append(close_score)
    
    return retval
#end word2vec_similarity

#------------------------------end similarity functions------------------------------

#run to get best match between first value (skillsforall) and all linkedin jobs, num_matches for how many matches to get
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
    return indexes


#return number of items above threshold
#pass in first row of similarity array
def num_passing(array, index):
    counter = 0
    for num in array:
        if num >= index:
            counter += 1
    return counter
#end num_passing

#------------------------------END FUNCTIONS------------------------------

#get word frequency NOT important to actual program
#start get word frequency
temp_data_word_freq = ''.join(data[1:])
linkedin_word_freq = get_word_freq(' '.join(word_preprocess(temp_data_word_freq))) #these are for analysis
skillsforall_word_freq = get_word_freq(' '.join(word_preprocess(data[0])))
#end get word frequency

#get similarity matrix and convert to readable format
similarity_cos = cos_similarity(data).toarray() #this variable is inefficient, translate sparse array to numpy array
print(similarity_cos[0,:])

similarity_word2vec = word2vec_similarity(data)
similarity_word2vec = [similarity_word2vec, [0]*len(similarity_word2vec)] #add bogus row to use with best_match
print(similarity_word2vec)

#similarity_jaccard = jaccard_similarity(data)
#print(similarity_jaccard) #returns 1D arary of similarities

print("----------")

print("closest jobs via tf-idf vectorization:")
#get closest matches by comparing first value in matrix (skillsforall) to values in first row (linkedin jobs)
best_jobs_cos = best_match(similarity_cos, 47)

#print results from best_jobs
for idx, job in enumerate(best_jobs_cos): #use enumerate to get index
    #add 1 to index to convert from programmer leetcode to normal person
    print("The #" + str(idx+1) + " closest job to this course is from " + str(job[0]) + ", as a " + str(job[1]) + ".")

print(str(num_passing(similarity_cos[0,1:], 0.05)) + " jobs passed similarity check, out of " + str(len(similarity_cos[0,1:])) + ".")

print("----------")

print("closest jobs via word2vec vectorization:")
#best closest matches with word2vec
best_jobs_word2vec = best_match(similarity_word2vec, 47)
for idx, job in enumerate(best_jobs_word2vec): #use enumerate to get index
    #add 1 to index to convert from programmer leetcode to normal person
    print("The #" + str(idx+1) + " closest job to this course is from " + str(job[0]) + ", as a " + str(job[1]) + ".")
numpy_similarity_word2vec = numpy.array(similarity_word2vec) #do this to clide in line below
print(str(num_passing(numpy_similarity_word2vec[0,1:], 0.05)) + " jobs passed similarity check, out of " + str(len(numpy.array(numpy_similarity_word2vec[0,1:]))) + ".")


#print("end program")

#TODO: get job links from linkedin & display alongside these
#TODO: output to file

#find a bunch of different ways to vectorize the text, start putting up slide with logic
#verify size of vectors are the same
#tensorflow word embeddinghello

#intro & overview (why are we doing this?)
#linknedin data grabbing
#skillsforall data grabbing
#text preprocessing, what do? alphanumeric, certain words, etc.
#text comparison (how? different models? similarity values?)
#how to rank similarity (what cutoff do we use? is it consistent between all different nlp models?)
#graphs of step before
#what's next? to continue the project

#histograph of scores from tfidf vs word2vec
#BERT
#tensorflow
#one more slide on results, then fun thing try to get another model (like bert)
#run datasets for larger grabs