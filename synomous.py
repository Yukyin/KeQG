import re
import numpy as np
from functools import reduce
from collections import OrderedDict
from nltk.corpus import wordnet
from nltk import word_tokenize, pos_tag
from nltk.tokenize.punkt import PunktSentenceTokenizer
from stanfordcorenlp import StanfordCoreNLP
from nltk.stem import WordNetLemmatizer
import os
# import ahocorasick
import spacy
from spacy.symbols import ORTH
import sentencepiece as spm
import gensim
import torch
# import benepar
import pickle
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
# from evidence_generator_bert_evaluation import evidenceEvaluator
# from nltk.tokenize import WordPunctTokenizer
from operator import itemgetter


lemmatizer = WordNetLemmatizer()
path = '/home/yukyin/download/stanford-corenlp-4.1.0'
corenlp = StanfordCoreNLP(path)
nlp = spacy.load('en_core_web_sm')


# glove
glove_ori="/home/yukyin/download/glove/glove.840B.300d.txt"
glove_w2v="/home/yukyin/download/glove/glove.840B.300d.w2v.txt"
glove_w2v_pkl="/home/yukyin/download/glove/glove.840B.300d.w2v.pkl"

if not os.path.exists(glove_w2v):
    glove2word2vec(glove_ori, glove_w2v)

if os.path.exists(glove_w2v_pkl):
    with open(glove_w2v_pkl, 'rb') as f:
        glove=pickle.load(f)
else:
    glove = gensim.models.KeyedVectors.load_word2vec_format(glove_w2v, binary=False)
    with open(glove_w2v_pkl,'wb') as f:
       pickle.dump(glove,f)




def is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

# def select_word_tokenize(sent):#选词用，这种方法不能保留句末标点，但是需要保留句中标点，比如 R&B, 需要保留&但不保留,
#     doc = nlp(sent)#把字符串转为文本，即去掉引号
#     # print([token.text for token in doc])
#     return [token.text for token in doc if not token.text in punctuation]#提取把文本中每个单词，把每个单词转为字符串放入list


def get_synonyms(word):
    '''
    Given a word, we retrieve its synonyms set by WordNet.
    '''
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(l.name())
    return list(set(synonyms))


def get_antonyms(word):
    """
    Given a word, we retrieve its antonyms set by WordNet.
    """
    antonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return list(set(antonyms))


def get_semantic_related_words(token, topN):
    """
    Given a word, we retrieve its topN most similar words list by Glove.
    """
    semantic_related = []
    if token in glove.vocab:
        token_in_glove = token
    elif token.lower() in glove.vocab:
        token_in_glove = token.lower()
    else:
        token_in_glove = None
    if token_in_glove is not None:
        semantic_related = glove.most_similar(positive=[token_in_glove], topn=topN)
        semantic_related = [item[0] for item in semantic_related]

    return semantic_related


def get_all_word_forms(word):
    """
        Given a word, get all its variant forms.
        For example, given "president", we can get:
        {'President', 'preside', 'presidency', 'Presidents',
        'presidencies', 'presidentships', 'presides', 'presided',
        'presidents', 'presiding', 'president', 'presidentship',
        'presidentially', 'presidential'}
        """
    forms = set()  # We'll store the derivational forms in a set to eliminate duplicates
    for lemma in wordnet.lemmas(word):  # for each "happy" lemma in WordNet
        forms.add(lemma.name())  # add the lemma itself
        for related_lemma in lemma.derivationally_related_forms():  # for each related lemma
            forms.add(related_lemma.name())  # add the related lemma
    forms.add(word)
    return list(forms)



# def find_sigword(text):
#     '''
#     find significant words in a text
#     '''
#     # clean_text = re.sub(u"\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", text)#去除括号中的文本

#     # pos = corenlp.pos_tag(clean_text)
#     tokens = corenlp.word_tokenize(text)
#     # tokens = select_word_tokenize(text)  # 分词
#     pos = nltk.pos_tag(tokens)  # 词性标注
#     sigword = [i for (i, j) in pos if not j in nonsig_pos_list and not i.lower() in question_symbols_list]
#     sigword=set(sigword)
#     return sigword


def get_wordnet_pos(tag):
    '''
    get pos of a word
    '''
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return None

def lemmatization(token):
    '''
    lemmatization of a word
    '''
    try:
        tagged_token = pos_tag([token])[0]     # 获取单词词性
        wnl = WordNetLemmatizer()
        word_pos = get_wordnet_pos(tagged_token[1]) or wordnet.NOUN
        lemmas_word=wnl.lemmatize(tagged_token[0], pos=word_pos)# 词形还原
        return lemmas_word
    except:#分词错误把一些非单词给分出来了
        # print(token)
        return token


def find_related_word(sig_quesword, topN):
    '''
    find topN related words in wordnet of a word
    '''
    related_words_all= {}
    for token in sig_quesword:
        lemma_token=lemmatization(token)
        all_word_forms = get_all_word_forms(lemma_token)
        synonyms = get_synonyms(lemma_token)
        antonyms = get_antonyms(lemma_token)
        semantic_related = get_semantic_related_words(lemma_token, topN)
        related_words = {}
        related_words["all_words"] = set(all_word_forms + synonyms + antonyms + semantic_related)
        related_words["all_word_forms"] = all_word_forms
        related_words["synonyms"] = synonyms
        related_words["antonyms"] = antonyms
        related_words["semantic_related"] = semantic_related
        related_words_all[token]=related_words
    # print(related_words_all)
    return related_words_all


print(find_related_word(['appear'],10))