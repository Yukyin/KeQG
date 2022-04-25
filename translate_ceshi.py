# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 15:08
# @Author  : yukyin
# Talk is cheap, show me the code.

import requests
from bs4 import BeautifulSoup
from google_trans_new import google_translator


def Trans_Trans(word):
    # headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36 SE 2.X MetaSr 1.0'}
    r = requests.get('https://www.merriam-webster.com/dictionary/{}'.format(word))
    bs = BeautifulSoup(r.text,'lxml')
    DefEng = []
    for i in bs.find_all('span','dtText'):
        DefEng.append(i.text[2::])
    translator = google_translator()
    DefChn = []
    for i in range(len(DefEng)):
        DefChn.append(translator.translate(DefEng[i],lang_tgt='zh'))
    DefEng.extend(DefChn)
    return DefEng


while True:
    word = input('请输入查询的单词/输入“886”离开：')
    if word == '886':
        break
    Trans = Trans_Trans(word)
    if Trans:
        word_half_len = int(len(Trans)/2)
        for i in range(word_half_len):
            print(Trans[i])
            print(Trans[i+word_half_len])
            print('====================')
        print('')
        print('')
    else:
        print('[Attention]单词输入有误/没有找到解释，请重试')
