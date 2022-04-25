'''1. wiki entity identification:tagme'''

from nltk.util import pr
import tagme
import logging
import sys
import os.path

# 标注的“Authorization Token”，需要注册才有
tagme.GCUBE_TOKEN = "3e784260-b9de-4e75-8fa9-1fd3105c5dcd-843339462"

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')

def Annotation_mentions(txt):
    """
    发现那些文本中可以是维基概念实体的概念
    :param txt: 一段文本对象，str类型
    :return: 键值对，键为本文当中原有的实体概念，值为该概念作为维基概念的概念大小，那些属于维基概念但是存在歧义现象的也包含其内
    """
    annotation_mentions = tagme.mentions(txt)
    print(annotation_mentions.mentions)
    dic = dict()
    for mention in annotation_mentions.mentions:
        try:
            dic[str(mention).split(" [")[0]] = str(mention).split("] lp=")[1]
        except:
            logger.error('error annotation_mention about ' + mention)
    return dic


def Annotate(txt, language="en", theta=0.1):
    """
    解决文本的概念实体与维基百科概念之间的映射问题
    :param txt: 一段文本对象，str类型
    :param language: 使用的语言 “de”为德语, “en”为英语，“it”为意语.默认为英语“en”
    :param theta:阈值[0, 1]，选择标注得分，阈值越大筛选出来的映射就越可靠，默认为0.1
    :return:键值对[(A, B):score]  A为文本当中的概念实体，B为维基概念实体，score为其得分
    """
    annotations = tagme.annotate(txt, lang=language)
    dic = dict()
    for ann in annotations.get_annotations(theta):
        print(ann.uri())
        try:
            A, B, score = str(ann).split(" -> ")[0], str(ann).split(" -> ")[1].split(" (score: ")[0], str(ann).split(" -> ")[1].split(" (score: ")[1].split(")")[0]
            dic[(A, B)] = score
        except:
            logger.error('error annotation about ' + ann)
    return dic

'''2. wiki description'''
import wikipedia

def entity_description(entity):
    summary=wikipedia.summary(entity)
    return summary


'''3. candidate knowledge:triples'''

import os
import pandas as pd
import re
import spacy
from spacy.attrs import intify_attrs

nlp = spacy.load("en_core_web_sm")

import neuralcoref

import networkx as nx
# import matplotlib.pyplot as plt

# nltk.download('stopwords')
from nltk.corpus import stopwords

all_stop_words = ['many', 'us', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
                  'today', 'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august',
                  'september', 'october', 'november', 'december', 'today', 'old', 'new']
all_stop_words = sorted(list(set(all_stop_words + list(stopwords.words('english')))))

abspath = os.path.abspath('')  ## String which contains absolute path to the script file
# print(abspath)
os.chdir(abspath)


### ==================================================================================================
# Tagger

def get_tags_spacy(nlp, text):
    doc = nlp(text)  # 生成词对象
    entities_spacy = []  # Entities that Spacy NER found
    for ent in doc.ents:  # doc.ents表示每个token的实体识别结果
        entities_spacy.append([ent.text, ent.start_char, ent.end_char, ent.label_])
    return entities_spacy


def tag_all(nlp, text, entities_spacy):
    if ('neuralcoref' in nlp.pipe_names):
        nlp.pipeline.remove('neuralcoref')
    neuralcoref.add_to_pipe(nlp)  # Add neural coref to SpaCy's pipe
    doc = nlp(text)
    return doc


def filter_spans(spans):
    # Filter a sequence of spans so they don't contain overlaps
    get_sort_key = lambda span: (span.end - span.start, span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
            seen_tokens.update(range(span.start, span.end))
    return result


def tag_chunks(doc):
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': 'ENTITY'}, string_store))


def tag_chunks_spans(doc, spans, ent_type):
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        string_store = doc.vocab.strings
        for span in spans:
            start = span.start
            end = span.end
            retokenizer.merge(doc[start: end], attrs=intify_attrs({'ent_type': ent_type}, string_store))


def clean(text):
    # 文本清理
    text = text.strip('[(),- :\'\"\n]\s*')
    text = text.replace('—', ' - ')
    text = re.sub('([A-Za-z0-9\)]{2,}\.)([A-Z]+[a-z]*)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)(\"\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.\/)(\w+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z]{3,}\.)([A-Z]+[a-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)
    text = re.sub('([[A-Z]{1}[[.]{1}[[A-Z]{1}[[.]{1}) ([[A-Z]{1}[a-z]{1,2} )', r"\g<1> . \g<2>", text, flags=re.UNICODE)
    text = re.sub('([A-Za-z0-9]{2,}\.)([A-Za-z]+)', r"\g<1> \g<2>", text, flags=re.UNICODE)

    text = re.sub('’', "'", text, flags=re.UNICODE)  # curly apostrophe
    text = re.sub('‘', "'", text, flags=re.UNICODE)  # curly apostrophe
    text = re.sub('“', ' "', text, flags=re.UNICODE)
    text = re.sub('”', ' "', text, flags=re.UNICODE)
    text = re.sub("\|", ", ", text, flags=re.UNICODE)
    text = text.replace('\t', ' ')
    text = re.sub('…', '.', text, flags=re.UNICODE)  # elipsis
    text = re.sub('â€¦', '.', text, flags=re.UNICODE)
    text = re.sub('â€“', '-', text)  # long hyphen
    text = re.sub('\s+', ' ', text, flags=re.UNICODE).strip()
    text = re.sub(' – ', ' . ', text, flags=re.UNICODE).strip()

    return text


def tagger(text):
    df_out = pd.DataFrame(
        columns=['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag',
                 'Start', 'End', 'Dependency'])
    corefs = []  # 保存所有指代的词
    text = clean(text)  # 文本清理

    nlp = spacy.load("en_core_web_sm")
    entities_spacy = get_tags_spacy(nlp, text)  # 获得每个token的实体识别结果
    # print("SPACY entities:\n", ([ent for ent in entities_spacy]), '\n\n')
    document = tag_all(nlp, text, entities_spacy)  # 融入共指消解工具
    # for token in document:
    #    print([token.i, token.text, token.ent_type_, token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx, token.idx+len(token)-1, token.dep_])

    ### Coreferences
    #
    if document._.has_coref:
        for cluster in document._.coref_clusters:
            main = cluster.main  # 共指的词
            for m in cluster.mentions:  # 所有指代的词（包括其本身）
                if (str(m).strip() == str(main).strip()):  # 如果是其本身，则跳过
                    continue
                corefs.append([str(m), str(main)])  # 将所有指代的词加入corefs列表
    tag_chunks(document)

    # chunk - somethin OF something 名词分块
    spans_change = []
    for i in range(2, len(document)):
        w_left = document[i - 2]
        w_middle = document[i - 1]
        w_right = document[i]
        if w_left.dep_ == 'attr':
            continue
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY' and (
                w_middle.text == 'of'):  # or w_middle.text == 'for'): #  or w_middle.text == 'with'
            spans_change.append(document[w_left.i: w_right.i + 1])
    tag_chunks_spans(document, spans_change, 'ENTITY')

    # chunk verbs with multiple words: 'were exhibited' 动词分块
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i - 1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i: w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: verb + adp; verb + part
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i - 1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'ADP' or w_right.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i: w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk: adp + verb; part  + verb
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i - 1]
        w_right = document[i]
        if w_right.pos_ == 'VERB' and (w_left.pos_ == 'ADP' or w_left.pos_ == 'PART'):
            spans_change_verbs.append(document[w_left.i: w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk verbs with multiple words: 'were exhibited'
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i - 1]
        w_right = document[i]
        if w_left.pos_ == 'VERB' and (w_right.pos_ == 'VERB'):
            spans_change_verbs.append(document[w_left.i: w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'VERB')

    # chunk all between LRB- -RRB- (something between brackets)
    start = 0
    end = 0
    spans_between_brackets = []
    for i in range(0, len(document)):
        if ('-LRB-' == document[i].tag_ or r"(" in document[i].text):
            start = document[i].i
            continue
        if ('-RRB-' == document[i].tag_ or r')' in document[i].text):
            end = document[i].i + 1
        if (end > start and not start == 0):
            span = document[start:end]
            try:
                assert (u"(" in span.text and u")" in span.text)
            except:
                pass
                # print(span)
            spans_between_brackets.append(span)
            start = 0
            end = 0
    tag_chunks_spans(document, spans_between_brackets, 'ENTITY')

    # chunk entities  两个实体相邻时，合并
    spans_change_verbs = []
    for i in range(1, len(document)):
        w_left = document[i - 1]
        w_right = document[i]
        if w_left.ent_type_ == 'ENTITY' and w_right.ent_type_ == 'ENTITY':
            spans_change_verbs.append(document[w_left.i: w_right.i + 1])
    tag_chunks_spans(document, spans_change_verbs, 'ENTITY')

    doc_id = 1
    count_sentences = 0
    prev_dep = 'nsubj'
    for token in document:
        if (token.dep_ == 'ROOT'):
            if token.pos_ == 'VERB':
                #  将pipeline的输出保存到csv，列名：['Document#', 'Sentence#', 'Word#', 'Word', 'EntityType', 'EntityIOB', 'Lemma', 'POS', 'POSTag', 'Start', 'End', 'Dependency']
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_,
                                           token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx,
                                           token.idx + len(token) - 1, token.dep_]
            else:
                df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_,
                                           token.ent_iob_, token.lemma_, token.pos_, token.tag_, token.idx,
                                           token.idx + len(token) - 1, prev_dep]
        else:
            df_out.loc[len(df_out)] = [doc_id, count_sentences, token.i, token.text, token.ent_type_, token.ent_iob_,
                                       token.lemma_, token.pos_, token.tag_, token.idx, token.idx + len(token) - 1,
                                       token.dep_]

        if (token.text == '.'):
            count_sentences += 1
        prev_dep = token.dep_

    return df_out, corefs


### ==================================================================================================
### triple extractor

def get_predicate(s):
    pred_ids = {}
    for w, index, spo in s:
        if spo == 'predicate' and w != "'s" and w != "\"":  # = 11.95
            pred_ids[index] = w
    predicates = {}
    for key, value in pred_ids.items():
        predicates[key] = value
    return predicates


def get_subjects(s, start, end, adps):
    subjects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'subject' in spo or 'entity' in spo or 'object' in spo:
                subjects[index] = w
    return subjects


def get_objects(s, start, end, adps):
    objects = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'object' in spo or 'entity' in spo or 'subject' in spo:
                objects[index] = w
    return objects


def get_positions(s, start, end):
    adps = {}
    for w, index, spo in s:
        if index >= start and index <= end:
            if 'of' == spo or 'at' == spo:
                adps[index] = w
    return adps


def create_triples(df_text, corefs):
    ## 创建三元组
    sentences = []  # 所有句子
    aSentence = []  # 某个句子

    for index, row in df_text.iterrows():
        d_id, s_id, word_id, word, ent, ent_iob, lemma, cg_pos, pos, start, end, dep = row.items()
        if 'subj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'subject'])
        elif 'ROOT' in dep[1] or 'VERB' in cg_pos[1] or pos[1] == 'IN':
            aSentence.append([word[1], word_id[1], 'predicate'])
        elif 'obj' in dep[1]:
            aSentence.append([word[1], word_id[1], 'object'])
        elif ent[1] == 'ENTITY':
            aSentence.append([word[1], word_id[1], 'entity'])
        elif word[1] == '.':
            sentences.append(aSentence)
            aSentence = []
        else:
            aSentence.append([word[1], word_id[1], pos[1]])

    relations = []
    # loose_entities = []
    for s in sentences:
        if len(s) == 0: continue
        preds = get_predicate(s)  # Get all verbs
        """
        if preds == {}: 
            preds = {p[1]:p[0] for p in s if (p[2] == 'JJ' or p[2] == 'IN' or p[2] == 'CC' or
                     p[2] == 'RP' or p[2] == ':' or p[2] == 'predicate' or
                     p[2] =='-LRB-' or p[2] =='-RRB-') }
            if preds == {}:
                #print('\npred = 0', s)
                preds = {p[1]:p[0] for p in s if (p[2] == ',')}
                if preds == {}:
                    ents = [e[0] for e in s if e[2] == 'entity']
                    if (ents):
                        loose_entities = ents # not significant for now
                        #print("Loose entities = ", ents)
        """
        if preds:
            if (len(preds) == 1):
                # print("preds = ", preds)
                predicate = list(preds.values())[0]
                if (len(predicate) < 2):
                    predicate = 'is'
                # print(s)
                ents = [e[0] for e in s if e[2] == 'entity']
                # print('ents = ', ents)
                for i in range(1, len(ents)):
                    relations.append([ents[0], predicate, ents[i]])

            pred_ids = list(preds.keys())
            pred_ids.append(s[0][1])
            pred_ids.append(s[len(s) - 1][1])
            pred_ids.sort()

            for i in range(1, len(pred_ids) - 1):
                predicate = preds[pred_ids[i]]
                adps_subjs = get_positions(s, pred_ids[i - 1], pred_ids[i])
                subjs = get_subjects(s, pred_ids[i - 1], pred_ids[i], adps_subjs)
                adps_objs = get_positions(s, pred_ids[i], pred_ids[i + 1])
                objs = get_objects(s, pred_ids[i], pred_ids[i + 1], adps_objs)
                for k_s, subj in subjs.items():
                    for k_o, obj in objs.items():
                        obj_prev_id = int(k_o) - 1
                        if obj_prev_id in adps_objs:  # at, in, of
                            relations.append([subj, predicate + ' ' + adps_objs[obj_prev_id], obj])
                        else:
                            relations.append([subj, predicate, obj])

    ### Read coreferences: coreference files are TAB separated values
    coreferences = []
    for val in corefs:
        if val[0].strip() != val[1].strip():
            if len(val[0]) <= 50 and len(val[1]) <= 50:
                co_word = val[0]
                real_word = val[1].strip('[,- \'\n]*')
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])
            else:
                co_word = val[0]
                real_word = ' '.join((val[1].strip('[,- \'\n]*')).split()[:7])
                real_word = re.sub("'s$", '', real_word, flags=re.UNICODE)
                if (co_word != real_word):
                    coreferences.append([co_word, real_word])

    # Resolve corefs
    triples_object_coref_resolved = []
    triples_all_coref_resolved = []
    for s, p, o in relations:
        coref_resolved = False
        for co in coreferences:
            if (s == co[0]):
                subj = co[1]
                triples_object_coref_resolved.append([subj, p, o])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_object_coref_resolved.append([s, p, o])

    for s, p, o in triples_object_coref_resolved:
        coref_resolved = False
        for co in coreferences:
            if (o == co[0]):
                obj = co[1]
                triples_all_coref_resolved.append([s, p, obj])
                coref_resolved = True
                break
        if not coref_resolved:
            triples_all_coref_resolved.append([s, p, o])
    return (triples_all_coref_resolved)


def get_graph(triples):
    G = nx.DiGraph()
    for s, p, o in triples:
        G.add_edge(s, o, key=p)
    return G


def get_entities_with_capitals(G):
    entities = []
    for node in G.nodes():
        if (any(ch.isupper() for ch in list(node))):
            entities.append(node)
    return entities


def get_paths_between_capitalised_entities(triples):
    g = get_graph(triples)
    ents_capitals = get_entities_with_capitals(g)
    paths = []
    # print('\nShortest paths among capitalised words -------------------')
    for i in range(0, len(ents_capitals)):
        n1 = ents_capitals[i]
        for j in range(1, len(ents_capitals)):
            try:
                n2 = ents_capitals[j]
                path = nx.shortest_path(g, source=n1, target=n2)
                if path and len(path) > 2:
                    paths.append(path)
                path = nx.shortest_path(g, source=n2, target=n1)
                if path and len(path) > 2:
                    paths.append(path)
            except Exception:
                continue
    return g, paths


def get_paths(doc_triples):
    triples = []
    g, paths = get_paths_between_capitalised_entities(doc_triples)
    for p in paths:
        path = [(u, g[u][v]['key'], v) for (u, v) in zip(p[0:], p[1:])]
        length = len(p)
        if (path[length - 2][1] == 'in' or path[length - 2][1] == 'at' or path[length - 2][1] == 'on'):
            if [path[0][0], path[length - 2][1], path[length - 2][2]] not in triples:
                triples.append([path[0][0], path[length - 2][1], path[length - 2][2]])
        elif (' in' in path[length - 2][1] or ' at' in path[length - 2][1] or ' on' in path[length - 2][1]):
            if [path[0][0], path[length - 2][1], path[length - 2][2]] not in triples:
                triples.append([path[0][0], 'in', path[length - 2][2]])
    for t in doc_triples:
        if t not in triples:
            triples.append(t)
    return triples


def get_center(nodes):
    center = ''
    if (len(nodes) == 1):
        center = nodes[0]
    else:
        # Capital letters and longer is preferred
        cap_ents = [e for e in nodes if any(x.isupper() for x in e)]
        if (cap_ents):
            center = max(cap_ents, key=len)
        else:
            center = max(nodes, key=len)
    return center


def connect_graphs(mytriples):
    G = nx.DiGraph()
    for s, p, o in mytriples:
        G.add_edge(s, o, p=p)

    """
    # Get components
    graphs = list(nx.connected_component_subgraphs(G.to_undirected()))

    # Get the largest component
    largest_g = max(graphs, key=len)
    largest_graph_center = ''
    largest_graph_center = get_center(nx.center(largest_g))

    # for each graph, find the centre node
    smaller_graph_centers = []
    for g in graphs:        
        center = get_center(nx.center(g))
        smaller_graph_centers.append(center)

    for n in smaller_graph_centers:
        if (largest_graph_center is not n):
            G.add_edge(largest_graph_center, n, p='with')
    """
    return G


def rank_by_degree(mytriples):  # , limit):
    G = connect_graphs(mytriples)
    degree_dict = dict(G.degree(G.nodes()))
    nx.set_node_attributes(G, degree_dict, 'degree')

    # Use this to draw the graph
    # draw_graph_centrality(G, degree_dict)

    Egos = nx.DiGraph()
    for a, data in sorted(G.nodes(data=True), key=lambda x: x[1]['degree'], reverse=True):
        ego = nx.ego_graph(G, a)
        Egos.add_edges_from(ego.edges(data=True))
        Egos.add_nodes_from(ego.nodes(data=True))

        # if (nx.number_of_edges(Egos) > 20):
        #    break

    ranked_triples = []
    for u, v, d in Egos.edges(data=True):
        ranked_triples.append([u, d['p'], v])
    return ranked_triples



# 抽取三元组
def extract_triples(text):
    df_tagged, corefs = tagger(text)  # pipeline处理文本，并返回每个token的特征，以及共指消解的结果
    doc_triples = create_triples(df_tagged, corefs)
    all_triples = get_paths(doc_triples)
    filtered_triples = []
    for s, p, o in all_triples:
        if ([s, p, o] not in filtered_triples):
            if s.lower() in all_stop_words or o.lower() in all_stop_words:
                continue
            elif s == p:
                continue
            if s.isdigit() or o.isdigit():
                continue
            if '%' in o or '%' in s:  # = 11.96
                continue
            if (len(s) < 2) or (len(o) < 2):
                continue
            if (s.islower() and len(s) < 4) or (o.islower() and len(o) < 4):
                continue
            if s == o:
                continue
            subj = s.strip('[,- :\'\"\n]*')
            pred = p.strip('[- :\'\"\n]*.')
            obj = o.strip('[,- :\'\"\n]*')

            for sw in ['a', 'an', 'the', 'its', 'their', 'his', 'her', 'our', 'all', 'old', 'new', 'latest', 'who',
                       'that', 'this', 'these', 'those']:
                subj = ' '.join(word for word in subj.split() if not word == sw)
                obj = ' '.join(word for word in obj.split() if not word == sw)
            subj = re.sub("\s\s+", " ", subj)
            obj = re.sub("\s\s+", " ", obj)

            if subj and pred and obj:
                filtered_triples.append([subj, pred, obj])

    # TRIPLES = rank_by_degree(filtered_triples)
    return filtered_triples

'''4.entity intepretation'''
from stanfordcorenlp import StanfordCoreNLP
import nltk
import requests
from bs4 import BeautifulSoup

path = '/home/yukyin/download/stanford-corenlp-4.1.0'
corenlp = StanfordCoreNLP(path)
nlp = spacy.load('en_core_web_sm')
nonsig_pos_list=['CC','MD','PDT','POS','TO','UH','#','DT','IN','SYM','WDT','WP','WP$','WRB','"','"','-LRB-','-RRB-',',','.',':','PRP','PRP$']#去掉介词和that，介词和that使用dependency parse填补

def find_sigword(text):
    '''
    find significant words in a text
    '''
    # clean_text = re.sub(u"\\(.*?\\)|\\{.*?\\}|\\[.*?\\]|\\<.*?\\>", "", text)#去除括号中的文本

    # pos = corenlp.pos_tag(clean_text)
    tokens = corenlp.word_tokenize(text)
    # tokens = select_word_tokenize(text)  # 分词
    pos = nltk.pos_tag(tokens)  # 词性标注
    sigword = [i for (i, j) in pos if not j in nonsig_pos_list and not i.lower() in question_symbols_list]
    sigword_list=set(sigword)
    
    for word in sigword_list:
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





if __name__ == '__main__':
    txt="which time zone is guinea-bissau in?"
    obj = Annotation_mentions(txt)
    entity_dict={}
    for i in obj.keys():
        print(i + "  " + obj[i])
        entity_dict.update({i:float(obj[i])})
    print(entity_dict)
    
    entity_list=list(entity_dict.keys())
    print(entity_list)

    candidate_entity_list=[key for key,value in entity_dict.items() if value>=0.5]
    print(candidate_entity_list)

    for entity in candidate_entity_list:
        descrip=entity_description(entity)
        print(descrip)
        triples=extract_triples(descrip)
        print(triples)