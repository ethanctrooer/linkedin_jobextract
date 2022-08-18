from scipy import spatial
import gensim.downloader as api
import numpy as np

model = api.load("glove-wiki-gigaword-50") #choose from multiple models https://github.com/RaRe-Technologies/gensim-data

s0 = 'Mark zuckerberg owns the facebook company'
s1 = 'Facebook company ceo is mark zuckerberg'
s2 = 'Microsoft is owned by Bill gates'
s3 = 'How to learn japanese'

def preprocess(s):
    x = [i.lower() for i in s.split()]
    return x

def get_vector(s):
    str_as_array = []
    retval = []
    for value in s:
        
        test = [model[i] for i in preprocess(value)]
        test2 = []
        for i in preprocess(value):
            test2.append(model[i])
        
        str_as_array.append(np.sum(np.array([model[i] for i in preprocess(value)]), axis=0))
    #return retval
    for value in str_as_array:
        close_score = 1 - spatial.distance.cosine(str_as_array[0], value)
        print('s0 vs this',close_score)
        retval.append(close_score)
    return retval
s = [s0, s1, s2, s3]

print(get_vector(s))

#print('s0 vs s1 ->',1 - spatial.distance.cosine(get_vector(s0), get_vector(s1)))
#print('s0 vs s2 ->', 1 - spatial.distance.cosine(get_vector(s0), get_vector(s2)))
#print('s0 vs s3 ->', 1 - spatial.distance.cosine(get_vector(s0), get_vector(s3)))