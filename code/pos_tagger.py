import argparse
import collections
import math
import operator
import random
import datetime

import utils
import numpy as np

def create_model(sentences):
    ## Data structures to store the model.
    ## You can modify this data structures if you want to
    prior_counts = collections.defaultdict(lambda: collections.defaultdict(int))
#     priors = collections.defaultdict(lambda: collections.defaultdict(float))
    likelihood_counts = collections.defaultdict(lambda: collections.defaultdict(int))
#     likelihoods = collections.defaultdict(lambda: collections.defaultdict(float))

    majority_tags = collections.defaultdict(lambda: collections.defaultdict(int))
    majority_baseline = collections.defaultdict(lambda: "NN")

    Tags_countVal = collections.defaultdict(int)

    # TODO: Create the model (counts) for the majority baseline.
    #   This model only needs to store the most frequent tag for each word
    
    majority_baseline = majority_train_model(sentences)
    # TODO: Create te model for the HMM model
    #   You will need to return the prior and likelihood probabilities (and possibly other stuff).
    #   At the end of the day, you need to return what you need in Tags_Prediction(...)
    #   Decide what to smooth, whether you need log probabilities, etc.

    model_ls = train_model(sentences)
    priors, likelihoods, tags, vocabulary = decode_seq(model_ls)
    ## You can modify the return value if you want to
    return priors, likelihoods, majority_baseline, Tags_countVal, model_ls, vocabulary, tags

def majority_train_model(sentences):
    majority_tags = collections.defaultdict(lambda: collections.defaultdict(int))
    majority_wordtag = collections.defaultdict(lambda: "NN")
    
    for sentence in sentences:
        for token in sentence:
#             print(token.word)
            word = token.word
            tag = token.tag
            majority_tags[word][tag]+=1
    
    for word in majority_tags:
        max_count = float(-np.inf)
        for tag in majority_tags[word]:
            if max_count < majority_tags[word][tag]:
               max_count = majority_tags[word][tag]
               majority_wordtag[word] = tag
               
    return majority_wordtag
            
            
def train_model(sentences):
    EmissionMat = collections.defaultdict(dict)
    TransMat = collections.defaultdict(dict)
    Tags_countVal = collections.defaultdict(int)
    vocabulary = collections.defaultdict(int)
    
    prev = "<s>"
    
    for sentence in sentences:
        for token in sentence:
            word = token.word
            tag = token.tag
            if ((prev in TransMat) and (tag in TransMat[prev])):
               TransMat[prev][tag] += 1
            else:
               TransMat[prev][tag] = 1
            if ((tag in EmissionMat) and (word in EmissionMat[tag])):
               EmissionMat[tag][word] += 1
            else:
               EmissionMat[tag][word] = 1
            Tags_countVal[tag] +=1
            vocabulary[word]+=1
            prev = tag
    vocabulary = sorted(vocabulary)
     
    return EmissionMat, TransMat, Tags_countVal, vocabulary
    
def decode_seq(model):
    """
    Decode sequences
    """
    EmissionMat, TransMat, context, vocabulary = model
    tags = sorted(context.keys())

    
    Mat_A = construct_A(TransMat, context, tags, vocabulary)
    Mat_B = construct_B(EmissionMat, context, tags, vocabulary)
    
    return Mat_A,Mat_B, tags, vocabulary


def construct_A(TransMat, context, tags, vocabulary):
   
    K = len(tags)
    N = len(vocabulary) 
    for i in range(K):
        for j in range(K):
            prev = tags[i]
            tag = tags[j]
            
            count = 0
            if ((prev in TransMat) and (tag in TransMat[prev])):
                count = TransMat[prev][tag]
                TransMat[prev][tag] = count/context[prev]
            else:
                TransMat[prev][tag] = (count + 1) / (context[prev] + K)

    return TransMat


def construct_B(EmissionMat, context, tags, vocabulary):
   
    K = len(tags)
    N = len(vocabulary)
    for i in range(K):
        for j in range(N):
            tag = tags[i]
            word = vocabulary[j]
        
            count = 0
            if word in EmissionMat[tag]:
                count = EmissionMat[tag][word]
                EmissionMat[tag][word] = count/context[tag]
            else:
                EmissionMat[tag][word] = (count + 1) / (context[tag] + N)


    return EmissionMat
    
    
def Tags_Prediction(sentences, model, mode='always_NN'):
    priors, likelihoods, majority_baseline, Tags_countVal, model_ls, vocabulary, tags = model

    for sentence in sentences:
        if mode == 'always_NN':
            # Do NOT change this one... it is a baseline
            for token in sentence:
                token.tag = "NN"
        elif mode == 'majority':
            # Do NOT change this one... it is a (smarter) baseline
            for token in sentence:
                token.tag = majority_baseline[token.word]
        elif mode == 'hmm':
            # TODO The bulk of your code goes here
            #   1. Create the Viterbi Matrix
            #   2. Fill the Viterbi matrix
            #      You will need one loop to fill the first column
            #      and a triple nested loop to fill the remaining columns
            #   3. Recover the sequence of tags and update token.tag accordingly
            # The current implementation tags everything as an NN you need to change it
            tag_seq, unseen_data = viterbi_matrix(priors, likelihoods, sentence, tags, vocabulary)
            i=0
            for token in sentence:
                word = token.word
                if word=="<s>" or word == "</s>" :
                    continue
                if word in unseen_data:
                    token.tag = unseen_word(word)
                    continue 
#                 print(token.word, tag_seq[i])
                token.tag = tag_seq[i]
#                 print(token.tag, tag_seq[i])
                
                i+=1
        else:
            assert False

    return sentences

        
noun_pos = ["action", "age", "ance", "cy", "dom", "ee", "ence", "er", "hood", "ion", "ism", "ist", "ity", "ling", "ment", "ness", "or", "ry", "scape", "ship", "ty"]
verb_pos= ["ate", "ify", "ise", "ize", "ing"]
adjective_pos = ["able", "ese", "ful", "i", "ian", "ible", "ic", "ish", "ive", "less", "ly", "ous"]
adverb_pos = ["ward", "wards", "wise"]

def unseen_word(word):
    if word[0].isupper():
       return "NNP"
    elif any(char.isdigit() for char in word):
       return "CD"
    elif any(word.endswith(suffix) for suffix in adjective_pos):
       return "JJ"
    elif word.endswith("s"):
       return "NNS"
    elif any(word.endswith(suffix) for suffix in adverb_pos):
       return "RB"
    elif any(word.endswith(suffix) for suffix in verb_pos):
       return "VBG"
    
    else:
       return "NN"           
            
def viterbi_matrix(priors, likelihoods, sentence, tags, vocabulary):
    tags = tags[2:]
    T = len(tags)
    S = len(sentence)
    v = np.zeros(shape=(T+1, S+1))
    tag_mat = collections.defaultdict(dict)
    
    tag_seq = []
    j=0
    i=1
    unseen_data = []
    current_pos_tag = "<s>"
    for token in sentence:    
        word = token.word
        if word=="<s>" or word =="</s>":
           continue
        if word not in vocabulary:
           unseen_data.append(word)
           continue
			
        max_value = -1
        local_max_val = -1
        j=1
        current_pos_tag = "<s>"
        for tag in tags:
            if(i==1):
               local_max_val = likelihoods[tag][word] * priors["<s>"][tag]
               tag_mat[tag, i] = tag
            else:
                local_max_val = -1
                k =1
                for tag_loc in tags:
                    updated_prob = v[k, i-1]*likelihoods[tag][word]*priors[tag_loc][tag]
                    if(local_max_val < updated_prob):
                       local_max_val = updated_prob
                       prev_pos_tag = tag_loc
                    k+=1
                tag_mat[tag, i] = prev_pos_tag
                if(max_value<local_max_val):
                   max_value = local_max_val
                   prev_tag = prev_pos_tag
                   current_pos_tag = tag
                
            v[j, i] = local_max_val
            j += 1
        i +=1
    
    tag_loc = current_pos_tag     
    tag_seq.append(tag_loc)
    for j in reversed(range(2,i)):
        tag_seq.append(tag_mat[tag_loc, j])
        tag_loc = tag_mat[tag_loc, j]
    return tag_seq[::-1], unseen_data
    
if __name__ == "__main__":
    # Do NOT change this code (the main method)
    parser = argparse.ArgumentParser()
    parser.add_argument("PATH_TR",
                        help="Path to train file with POS annotations")
    parser.add_argument("PATH_TE",
                        help="Path to test file (POS tags only used for evaluation)")
    parser.add_argument("--mode", choices=['always_NN', 'majority', 'hmm'], default='always_NN')
    args = parser.parse_args()

    tr_sents = utils.read_tokens(args.PATH_TR) #, max_sents=1)
    # test=True ensures that you do not have access to the gold tags (and inadvertently use them)
    te_sents = utils.read_tokens(args.PATH_TE, test=True)

    model = create_model(tr_sents)

     print("** Testing the model with the training instances (boring, this is just a sanity check)")
     gold_sents = utils.read_tokens(args.PATH_TR)
     predictions = Tags_Prediction(utils.read_tokens(args.PATH_TR, test=True), model, mode=args.mode)
     accuracy = utils.calc_accuracy(gold_sents, predictions)
     print(f"[{args.mode:11}] Accuracy "
           f"[{len(list(gold_sents))} sentenceences]: {accuracy:6.2f} [not that useful, mostly a sanity check]")
     print()
 
     print("** Testing the model with the test instances (interesting, these are the numbres that matter)")
    
    # read sentenceences again because Tags_Prediction(...) rewrites the tags
    gold_sents = utils.read_tokens(args.PATH_TE)
    predictions = Tags_Prediction(te_sents, model, mode=args.mode)
    accuracy = utils.calc_accuracy(gold_sents, predictions)
    print(f"[{args.mode}:11] Accuracy "
          f"[{len(list(gold_sents))} sentenceences]: {accuracy:6.2f}")













