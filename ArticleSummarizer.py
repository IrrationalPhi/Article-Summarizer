#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan  1 00:10:05 2022

@author: phi
"""

# SETUP
import os
import pathlib
from collections import Counter
working_dir = pathlib.Path().absolute()
os.chdir(working_dir)

import nltk
import requests
import numpy as np
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
import tkinter as tk
from jenks import jenks

# CONSTANTS

ACRONYM_MULTIPLIER = 2
SENTENCE_START_MULTIPLIER = 2
FINANCIAL_MULTIPLIER = 2
TITLE_MULTIPLIER = 2
LENGTH_THRESHOLD = 50
NUMBER_MULTIPLIER = 2
GVF_THRESHOLD = 0.8
MIN_SENTENCES_TO_REPORT = 4

STOPWORDS = set(stopwords.words('english'))
FINANCIAL_WORDS = {
    '$', 'dollar', 'dollars', 'euro', 'euros',
    'peso', 'pesos', 'PHP',
}


def extract_text_info(url):
    """
    Extract the title and text from a news article through its URL.

    Parameters
    ----------
    url : String
        URL of news article to summarize.

    Returns
    -------
    text : String
        Body of news article.
    title_words : set
        Set of word tokens in article title.

    """
    
    # beautifulsoup setup
    page = requests.get(url).content
    soup = BeautifulSoup(page, "html.parser")
    
    # extracting main text
    paragraphs = soup.find_all('p')
    paragraph_texts = [paragraph.text.strip() for paragraph in paragraphs]
    paragraph_texts = [paragraph_text for paragraph_text in paragraph_texts if len(paragraph_text) >= LENGTH_THRESHOLD]
    text = " ".join(paragraph_texts)
    
    # extract possible important title words
    title = soup.find('h1').text.strip()
    
    # get rid of links + single characters + auxiliary stuff
    title_words = nltk.tokenize.word_tokenize(title, language = 'english')
    title_words = set([word.lower() for word in title_words if len(word) > 1])
    title_words = set([word for word in title_words if word not in STOPWORDS])
    
    return text, title_words
    

def get_word_scores(text, title_words):
    """
    Get word scores of each word in a text.

    Parameters
    ----------
    text : String
        Article main text to summarize.
    title_words : set
        Set of title words of article.

    Returns
    -------
    word_scores : Counter
        Counter of word tokens and corresponding scores. Only considers words
        of length at least 2 and are not stopwords.

    """
    
    
    # counter for words
    words_list = nltk.tokenize.word_tokenize(text, language = 'english')
    # only count words of length > 1 (to remove punctuations and other auxiliary stuff)
    # remove stop words from the list
    words_list = [word for word in words_list if len(word) > 1 and word.lower() not in STOPWORDS]
    # count number of word occurences
    # intuition: more frequent words are more important if they are not stop words
    word_scores = Counter(words_list)
    
    for word in word_scores:
        # if acronym (or strong emphasis), probably important
        if word.isupper():
            word_scores[word] *= ACRONYM_MULTIPLIER
        # if start of sentence, it's underrepresented
        # will only consider if it's long enough, say more than 3 letters
        elif word[0].isupper() and len(word) > 3:
            word_scores[word] *= SENTENCE_START_MULTIPLIER
        
        # if word important from title, add a little bonus
        if word.lower() in title_words:
            word_scores[word] *= TITLE_MULTIPLIER
        
        if word.isdigit():
            word_scores[word] *= NUMBER_MULTIPLIER
            
        # if financial word, may be important
        if word.lower in FINANCIAL_WORDS:
            word_scores[word] *= FINANCIAL_MULTIPLIER
            
    return word_scores

# get sentence score
def get_sentence_score(sentence, word_scores):
    """
    Obtain the score of a sentence through the scores of each word.

    Parameters
    ----------
    sentence : String
        Sentence whose score is to be computed.
    word_scores : Counter/dict
        Words and their corresponding scores.

    Returns
    -------
    temp_score : float
        The score of the sentence.

    """
    
    temp_word_list = nltk.tokenize.word_tokenize(sentence, language = 'english')
    temp_score = 0
    
    # for each word in sentence, we sum its score to get the sentence score
    for word in temp_word_list:
        if word in word_scores:
            temp_score += word_scores[word]
            
    return temp_score

def goodness_of_variance_fit(values, breakpoints):
    """
    Compute the goodness of variance fit (GVF) of a 
    1-dimensional clustering given array and breakpoints

    Parameters
    ----------
    values : float array
        Array of floats which we want to cluster.
    breakpoints : float array
        (Sorted) array of floats representing breakpoints.

    Returns
    -------
    res : float
        goodness of variance fit (between 0 and 1).
    zone_indices : array
        cluster numbers of each index

    """
    
    def classify(value, breakpoints):
        for i in range(1, len(breakpoints)):
            if value < breakpoints[i]:
                return i
        return len(breakpoints) -1
    
    values = np.array(values)
    # get cluster number of each element
    clusters = np.array([classify(value, breakpoints) for value in values])
     
    # aggregate per cluster, storing indices
    zone_indices = [[idx for idx, val in enumerate(clusters) 
                     if zone + 1 == val] for zone in range(len(breakpoints)-1)]
    
    # corresponding values
    zone_values = [np.array([values[idx] for idx in zone]) 
                   for zone in zone_indices]
    
    ssd_array_mean = np.sum((values - values.mean())**2)
    ssd_class_means = sum([np.sum((cluster - cluster.mean()) ** 2) 
                           for cluster in zone_values])
    
    res = (ssd_array_mean - ssd_class_means)/ssd_array_mean
    return res, zone_indices
    
def find_best_cluster(values, threshold):
    """
    Determines passable clustering given array and threshold via GVF.

    Parameters
    ----------
    values : array
        Array whose elements are to be clustered.
    threshold : float (between 0 and 1)
        GVF threshold for passable clustering.

    Returns
    -------
    zone_indices : array (length 2)
        Indices of top 2 clusters based on score.

    """
    
    
    num_clusters = 1
    goodness_variance_fit = 0.0
    while goodness_variance_fit < threshold:
        num_clusters += 1
        breakpoints = jenks(values, num_clusters)
        goodness_variance_fit, zone_indices = goodness_of_variance_fit(values, breakpoints)
           
    # possibly the second to last if too little
    return zone_indices[-2:]
        

def get_top_sentences(word_scores, sentences):
    """
    Retrieve top sentences of text via clustering.

    Parameters
    ----------
    word_scores : Counter/dict
        Words and their corresponding scores.
    sentences : List
        List of (ordered) sentences in article text..

    Returns
    -------
    res : List
        List of top sentences.

    """
    
    # first compute each sentence score
    sentence_scores = []
    for idx in range(len(sentences)):
        sentence = sentences[idx]
        score = get_sentence_score(sentence, word_scores)
        
        sentence_scores.append([sentence, score, idx])
        
    raw_scores = [el[1] for el in sentence_scores]
    top_two_clusters = find_best_cluster(raw_scores, GVF_THRESHOLD)
    top_sentence_idxs = top_two_clusters[1]
    second_sentence_idxs = top_two_clusters[0]
    
    top_sentences = [sentence_scores[idx] for idx in top_sentence_idxs]
    second_sentences = [sentence_scores[idx] for idx in second_sentence_idxs]
        
    # we want at least 4 sentences or something similar
    if len(top_sentence_idxs) < MIN_SENTENCES_TO_REPORT:
        top_sentences = top_sentences + second_sentences
    
    # sort top sentences by index
    top_sentences.sort(key = lambda x: x[2])
    
    res = [s[0] for s in top_sentences]
    
    return res

def line_breaker(text, max_length):    
    """
    Modify summary so that output does not cut in the middle of words.

    Parameters
    ----------
    text : String
        String to modify.
    max_length : int
        Desired maximum length of each output line.

    Returns
    -------
    res : String
        Modified version with optimal line breaks added.

    """
    
    
    words = text.split(" ")
    n = len(words)

    # initialize stuff for dp bookkeeping
    sum_lengths = [0]*(n+1)
    word_lengths = [0]*n

    # add 0 and "" for base case
    res_vals = [0]*(n+1)
    res_text = ['']*(n+1)

    temp_sum = 0
    for i in range(len(words)):
        temp = len(words[i]) + 1
        temp_sum += temp
        word_lengths[i] = temp
        sum_lengths[i+1] = temp_sum

    max_length += 1
    min_penalty = max_length**3 +1

    for i in range(len(words)):
        min_temp_val = min_penalty
        temp_str = ""
        for j in range(i+1):
            last_line = sum_lengths[i+1]- sum_lengths[j]
            temp = res_vals[j] + (max_length - last_line)**3

            if last_line <= max_length and temp < min_temp_val:
                min_temp_val = temp
                temp_str = res_text[j] + '\n' + text[sum_lengths[j]:sum_lengths[i+1]-1]

        res_vals[i+1] = min_temp_val
        res_text[i+1] = temp_str

    best_val = min_penalty
    res = ""
    for i in range(len(words)):
        if sum_lengths[len(words)] - sum_lengths[i] <= max_length and res_vals[i] < best_val:
            best_val = res_vals[i]
            res = res_text[i] + '\n' + text[sum_lengths[i]:]

    return res

def summarize():
    """
    Obtain URL string from textbox in tkinter interface and output
    summary of article in URL with reduction factor.

    Returns
    -------
    summary : String
        Summary of article.
    reduction_factor : float
        How much the original article was reduced in length.

    """
    
    
    url = url_box.get("1.0","end-1c")
    text, title_words = extract_text_info(url)
    word_scores = get_word_scores(text, title_words)

    # now get score per sentence
    # we will only look at sentences that are long enough (say > 100 characters)
    sentences = nltk.tokenize.sent_tokenize(text, language = 'english')
    sentences = [s for s in sentences if len(s) >= LENGTH_THRESHOLD]

    top_sentences = get_top_sentences(word_scores, sentences)
    summary = " ".join(top_sentences)
    reduction_factor = round(100*(1-len(summary)/len(text)),2)

    ## for line breaks
    summary = line_breaker(summary, 100)
    
    reduction_output = "\n\nReduction Factor: {reduction}%".format(reduction = reduction_factor)
    
    output = summary + reduction_output
    
    summary_box.delete('1.0', tk.END)
    summary_box.insert(tk.END, output)
    
    return summary, reduction_factor


# INTERFACE FOR USER
root = tk.Tk()
root.title('Article Summarizer')
root.geometry('1200x600')

# summary stuff
summary_label = tk.Label(root, text="Summary")
summary_label.pack()

summary_box = tk.Text(root, height=20, width=140)
summary_box.pack()

# url stuff
url_label = tk.Label(root, text="Enter URL")
url_label.pack()

url_box = tk.Text(root, height=1, width=140)
url_box.pack()

# to summarize
summarize_button =tk.Button(root, text='Summarize', command = summarize)
summarize_button.pack()

root.mainloop()