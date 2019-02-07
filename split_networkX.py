import os
import glob
import re
import numpy as np
import pandas as pd
import spacy
import matplotlib.pyplot as plt
import networkx as nx
import sys
from importlib import reload
from functools import reduce
reload(sys)
# from spacy.lang.en.stop_words import STOP_WORDS
import nltk.data

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
plt.rcParams['figure.figsize'] = [10, 10]
nlp = spacy.load('en')


line_items = ['Revenues', 'Assets', 'Expenses','sales']


metrics = ['increase', 'increased', 'decrease', 'decreased', 'grows', 'growth', 'declined', 'decline',
           'gains', 'gained', 'losses', 'lost', 'loses']

mda_path = '/home/saheli/Desktop/NLP/CITI/MDA'
text_path = '/home/saheli/Desktop/NLP/CITI/MDA/MDA_txt'

for filepath in glob.glob(os.path.join(mda_path,'*.mda')):
    with open(filepath) as f:
        file = f.readlines()  # .replace('\n', ' ')
        file_list = []
        for line in file:
            line = line.replace('\n', ' ')
            no_period_words = re.sub(r'(?<!\w)([A-Z])\.', r'\1', line)
            cleaned_lines = re.sub(r'\.+', '.', no_period_words)

            file_list.append(cleaned_lines)

        clean_file = ''.join(file_list)

        doc = '\n\n'.join(tokenizer.tokenize(clean_file))

        # replacement = [['sg&a expense','sga_expense'],['net current assets','net_current_assets'],['r&d expense','rd_expense']]
        # for ijx in replacement:
        #     doc = doc.lower().replace(ijx[0],ijx[1])

        file_out = open(mda_path+'/'+'MDA_txt/'+f.name.split('/')[-1].split('.')[0]+'.txt', 'w')
        file_out.write(doc)


def extract_child_head(docutf):
    all = [(token.text, token.pos_, token.dep_, token.head.text,  token.head.pos_, [child for child in token.children]) for token in docutf if token.is_punct != True]
    all = pd.DataFrame(all, columns = ['Text','Text-POS','DEP',  'Head', 'Head-POS', 'Children'])
    return all


def remove_stopwords(sentence):
    document = nlp(sentence.lower())
    clean_text =''.join(token.string for token in [word for word in document if  word.is_stop != True])
    return clean_text


def process(string):
    lists__ = ['increase', 'increased', 'decrease', 'decreased', 'grows', 'growth', 'declined', 'decline',
               'gain', 'gained', 'losses', 'lost', 'loses']
    return [k for k in lists__ if string.__contains__(k)]


for filepath in glob.glob(os.path.join(text_path,'*.txt')):
    file = [line.lower() for line in open(filepath).readlines()]
    required_line = [line for text in line_items for line in file if text.lower() in line]
    required_lines = list(map(lambda x: remove_stopwords(x), required_line))
    data = pd.DataFrame(required_lines, columns=['Lines'])
    sent = []
    tags = []
    l_tag_dict =[]
    for each in range(data.shape[0]):
        doc = extract_child_head(nlp(data.iloc[each][0]))
        G = nx.Graph()
        G = nx.from_pandas_edgelist(doc, 'Text', 'Head')
        line_items_adjlist = [[list(value.keys()), key] for key, value in G._adj.items() if
                              key in "|".join([item.lower() for item in line_items])]
        # line_items_adjlist_t = [(line_items_adjlist[0][1], y[0][0]) for y in line_items_adjlist if y!='']

        line_items_adjlist_ = []
        for elem in line_items_adjlist:
            tgs = [(elem[1], y) for y in elem[0]]
            line_items_adjlist_.append(tgs)
        line_items_adjlist_t = reduce(lambda x, y: x + y, line_items_adjlist_)

        r_tag = [elem for elem in list(map(process, [idx[1] for idx in line_items_adjlist_t])) if len(elem) > 0]
        if r_tag:
            line_tag = [elem for tag in r_tag[0] for elem in line_items_adjlist_t if tag in elem]
            line_tag_str = list(map(lambda x: x[0] + '_' + x[1], line_tag))
            lineitem_tag_dict = dict(zip(line_tag_str, line_tag))
        else:
            line_tag = None
            line_tag_str = None
            lineitem_tag_dict = None

        tags.append(line_items_adjlist_t)
        sent.append(data.iloc[each][0])
        l_tag_dict.append(lineitem_tag_dict)
    tags_sent = pd.DataFrame({'Sentence': sent, 'Tags': tags, 'LineItems_Tag dict': l_tag_dict})
    tags_sent.to_csv('/home/saheli/Desktop/NLP/CITI/MDA/MDA_Tags/'+filepath.split('/')[-1].split('.')[0]+'_Tags.csv')
    # req_tags_df = pd.DataFrame({'Required Tags' : required_tags})
    # req_tags_df.to_csv('/home/saheli/Desktop/NLP/CITI/MDA/MDA_Tags/Req_Tags/'+filepath.split('/')[-1].split('.')[0]+'_req_tags.csv')

# path = '/home/saheli/Desktop/NLP/CITI/MDA/MDA_Tags/'
# files = os.listdir(path)
#
# for ijx in range(len(files)):
#     if files[ijx].__contains__('.csv'):
#         selected_file = files[ijx]
#
#         csv = pd.read_csv(path+selected_file)
#
#         def process(string):
#             lists__ = ['increase', 'increased', 'decrease', 'decreased', 'grows', 'growth', 'declined', 'decline',
#                    'gain', 'gained', 'losses', 'lost', 'loses']
#             return [k for k in lists__ if string.__contains__(k)]
#
#         required_tags = []
#
#         for each in  range(csv.shape[0]):
#
#             cleaned_list = [idx for idx in list(map(process,csv.iloc[each,-1].replace("[",'').replace("]",'').replace(")",'').split(", ("))) if idx != []]
#             tuplelist = [k.replace("(",'') for  k in csv.iloc[each,-1].replace("[",'').replace("]",'').replace(")",'').split(", (") for j in cleaned_list if k.__contains__(j[0])]
#             extracted_metric_list = [j.replace("\'",'').split(", ")[0]+"_"+j.replace("\'",'').split(", ")[1] for j in tuplelist]
#             required_tags.append(extracted_metric_list)
#         if ijx == 0:
#             df1 = pd.DataFrame(required_tags).dropna()
#             df1.insert(0,"CO_NAME",np.repeat(selected_file.split("_Tags.csv")[0],df1.shape[0]))
#         else:
#             df2 = pd.DataFrame(required_tags).dropna()
#             df2.insert(0, "CO_NAME", np.repeat(selected_file.split("_Tags.csv")[0], df2.shape[0]))
#             df1 = pd.concat([df1,df2])
