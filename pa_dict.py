# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 14:14
# @Author  : yukyin
# Talk is cheap, show me the code.
import requests
from bs4 import BeautifulSoup

# get word from Command line
# word = input("Enter a word (enter 'q' to exit): ")
word='track'
# main body
while word != 'q': # 'q' to exit
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}
        # 利用GET获取输入单词的网页信息
        r = requests.get(url='https://dictionary.cambridge.org/dictionary/english/%s'%word,headers=headers)
        # 利用BeautifulSoup将获取到的文本解析成HTML
        soup = BeautifulSoup(r.text, "lxml")
        # 获取字典的标签内容
        DefEng = []
        for i in soup.find_all('ul'):
            DefEng.append(i.text[2::])
        print(DefEng)
    except Exception:
        print("Sorry, there is a error!\n")
    finally:
        word = input( "Enter a word (enter 'q' to exit): ")