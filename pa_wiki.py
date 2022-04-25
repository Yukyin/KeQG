# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 16:38
# @Author  : yukyin
# Talk is cheap, show me the code.
import requests

# proxy = {
#     "http": "http://127.0.0.1:58591",
#     "https": "https://127.0.0.1:58591"
# }
# data = requests.get("http://www.google.com.hk", proxies=proxy)
import urllib3
z = urllib3.connection_from_url('http://en.wikipedia.org/wiki/Main_Page')
z.request('GET', 'NOTAREALSCHEME://en.wikipedia.org/wiki/Main_Page', assert_same_host=False)



import wikipedia


summary=wikipedia.summary("guinea-bissau")
print(summary)
#"Ellen Henrietta Swallow Richards (December 3, 1842 – March 30, 1911) was an industrial and safety engineer, environmental chemist, and university faculty member in the United States during the 19th century. Her pioneering work in sanitary engineering, and experimental research in domestic science, laid a foundation for the new science of home economics. She was the founder of the home economics movement characterized by the application of science to the home, and the first to apply chemistry to the study of nutrition.Richards graduated from Westford Academy (second oldest secondary school in Massachusetts) in 1862. She was the first woman admitted to the Massachusetts Institute of Technology. She graduated in 1873 and later became its first female instructor. Mrs. Richards was the first woman in America accepted to any school of science and technology, and the first American woman to obtain a degree in chemistry, which she earned from Vassar College in 1870.Richards was a pragmatic feminist, as well as a founding ecofeminist, who believed that women's work within the home was a vital aspect of the economy. At the same time, however, she did not directly challenge the prevailing cult of domesticity that valorized women's place and work in the home."

# summary_part=wikipedia.summary("ellen swallow richards", sentences=1)
#
# search=wikipedia.search("ellen swallow richards")#相关词条，不需要
# page = wikipedia.page("ellen swallow richards")
# title=page.title#"ellen swallow richards"
# url=page.url
# content=page.content#全部内容
# links=page.links



