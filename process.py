# -*- coding: utf-8 -*-
# @Time    : 2021/6/15 19:30
# @Author  : yukyin
# Talk is cheap, show me the code.
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


'''三元组这一部分要用python37，spacy2.1，neurocoef4.0'''
import os
# import pandas as pd
import csv

from triples_from_text import extract_triples

triples = extract_triples("Guinea-Bissau, officially the Republic of Guinea-Bissau, is a country in West Africa that covers 36,125 square kilometres with an estimated population of 1,874,303. It borders Senegal to the north and Guinea to the south-east.Guinea-Bissau was once part of the kingdom of Kaabu, and part of the Mali Empire. Parts of this kingdom persisted until the 18th century, while a few others were under some rule by the Portuguese Empire since the 16th century. In the 19th century, it was colonised as Portuguese Guinea. Upon independence, declared in 1973 and recognised in 1974, the name of its capital, Bissau, was added to the country's name to prevent confusion with Guinea. Guinea-Bissau has a history of political instability since independence, and only one elected president (José Mário Vaz) has successfully served a full five-year term. The current president is Umaro Sissoco Embaló, who was elected on 29 December 2019.Only about 2% of the population speaks Portuguese, the official language, as a first language, and 33% speak it as a second language. However, Guinea-Bissau Creole is the national language and also considered the language of unity. According to a 2012 study, 54% of the population speak Creole as a first language and about 40% speak it as a second language. The remainder speak a variety of native African languages. There are diverse religions in Guinea-Bissau. Christianity and Islam are the main religions practised in the country. The country's per-capita gross domestic product is one of the lowest in the world.Guinea-Bissau is a member of the United Nations, African Union, Economic Community of West African States, Organisation of Islamic Cooperation, Community of Portuguese Language Countries, Organisation internationale de la Francophonie, and the South Atlantic Peace and Cooperation Zone, and was a member of the now-defunct Latin Union.")
print("\n\n===============the result=============\n\n")
print(triples)


# def process_all():
#     while(True):
#         # text = input("input a text:")
#         triples = extract_triples("It borders Senegal to the north and Guinea to the south-east.Guinea-Bissau was once part of the kingdom of Kaabu, as well as part of the Mali Empire.")
#         print("\n\n===============the result=============\n\n")
#         print(triples)

# # Reads data file and creates the submission.csv file
# if __name__ == "__main__":
#     process_all()
#     print("Finished the process.")

