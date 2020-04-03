# -*- coding: utf-8 -*-
import sys, os, multiprocessing, warnings, shutil, re, time, json
warnings.filterwarnings(action="ignore")

#현재 파일 경로
#main_dir = os.path.dirname(os.path.realpath(__file__))                # Python Script 파일 실행 시
main_dir = os.getcwd()                                                 # jupyter notebook에서 테스트 시
main_dir = [folder for folder in main_dir.split(os.path.sep) if folder.strip()]
main_dir = '/' + '/'.join(main_dir)
sys.path.append(main_dir)

import numpy as np, pandas as pd
from pandas import DataFrame
from joblib import Parallel, delayed
from sqlalchemy import create_engine
import fasttext
from chatspace import ChatSpace
from PyKomoran import *
import hanja
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from PIL import Image
from wordcloud import WordCloud, ImageColorGenerator
import matplotlib.pyplot as plt


host, database, user, passwd = '192.168.198.245:5432', 'crawlerdb', 'sycho', 'nics1995'
num_cores = int(multiprocessing.cpu_count()/2)

def lang_gbn(comment, lang_gbn_model):
    comment = comment.replace('\n', ' ').strip()                                                        # 개행 문자 -> 띄어쓰기 변경
    if bool(re.match(r'^[^a-zA-Z0-9가-힣]+$', comment)): result = (comment, 'unknown', 0)               # 정규식 기반 필터
    elif bool(len(comment) >= 5 and len(set(comment)) > 1) == False: result = (comment, 'unknown', 0) # 5자 이상 and 2개 문자 이상 사용
    else:
        result = lang_gbn_model.predict(comment)
        result = (comment, result[0][0].replace('__label__', '').strip(), round(result[1][0], 5))
    return list(result)

lang_gbn_model = os.path.join(main_dir, 'dictionary', 'lid.176.bin')
lang_gbn_model = fasttext.load_model(lang_gbn_model)

## Data

host, database, user, passwd = '192.168.198.245:5432', 'crawlerdb', 'sycho', 'nics1995'

def select_data(query):
    engine = create_engine('postgresql+psycopg2://{0}:{1}@{2}/{3}'.format(user, passwd, host, database), client_encoding='utf8')
    conn = engine.raw_connection()
    query_result  = pd.read_sql(query, conn)
    conn.close()
    return query_result

# data = select_data("""
# SELECT * FROM crawling_data.youtube_data
#     WHERE "category" = '펭수'
#     ORDER by random()
#     LIMIT 100
# ;""")

# data = {
#            'kingdom': '-',
#            'Castle': '-',
#            'id' : 'test_0001',
#            'text' : list(data.comments)
#          }
# data = json.dumps(data)
# data = json.loads(data)

# data = [lang_gbn(comment, lang_gbn_model) for comment in data['text']]
# data = DataFrame(data, columns = ['comment', 'lang', 'prob'])
#
# data = data[(data['lang'].isin(['ko', 'en'])) & (data['prob'] > 0.9)].drop(['prob'], axis = 1).reset_index(drop = True)
# ko_data = list(data[data.lang == 'ko'].comment)

f = open('test_comments.txt', 'r', encoding='UTF-8')
ko_data = f.read().strip().split('\n')
print(ko_data)

## Update

f = open(os.path.join(main_dir, 'dictionary', 'user.dic'), mode='r', encoding='utf-8')
reorder_user_dic = list(f)

for i, _ in enumerate(range(len(reorder_user_dic))):
    if not reorder_user_dic[i].endswith('\n'):
        reorder_user_dic[i] = reorder_user_dic[i] + '\n'
reorder_user_dic = sorted(reorder_user_dic, key=len, reverse=True)
f.close()

f = open(os.path.join(main_dir, 'dictionary', 'user.dic'), mode='w', encoding='utf-8')
f.writelines([line for line in reorder_user_dic])
f.close()
del reorder_user_dic

# 치환 사전 및 Tokenizer Load
f = open(os.path.join(main_dir, 'dictionary', 'replace_eomi.dic'), mode='r', encoding='utf-8')
replace_eomi = f.read().strip().split('\n')
replace_eomi = [line.split('\t') for line in replace_eomi if line.strip()[0] != '#']
replace_eomi = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in replace_eomi])
f.close()

f = open(os.path.join(main_dir, 'dictionary', 'replace_wrd.dic'), mode='r', encoding='utf-8')
replace_wrd = f.read().strip().split('\n')
replace_wrd = [line.split('\t') for line in replace_wrd if line.strip()[0] != '#']
f.close()

f = open(os.path.join(main_dir, 'dictionary', 'post_processor_1.dic'), mode='r', encoding='utf-8')
post_proc1 = f.read().strip().split('\n')
post_proc1 = [line.split('\t') for line in post_proc1 if line.strip()[0] != '#']
post_proc1 = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in post_proc1])
check_key1 = list(post_proc1.keys())
check_values1 = list(post_proc1.values())
f.close()

f = open(os.path.join(main_dir, 'dictionary', 'post_processor_2.dic'), mode='r', encoding='utf-8')
post_proc2 = f.read().strip().split('\n')
post_proc2 = [line.split('\t') for line in post_proc2 if line.strip()[0] != '#']
post_proc2 = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in post_proc2])
check_key2 = list(post_proc2.keys())
check_values2 = list(post_proc2.values())
f.close()

f = open(os.path.join(main_dir, 'dictionary', 'post_processor_3.dic'), mode='r', encoding='utf-8')
post_proc3 = f.read().strip().split('\n')
post_proc3 = [line.split('\t') for line in post_proc3 if line.strip()[0] != '#']
post_proc3 = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in post_proc3])
check_key3 = list(post_proc3.keys())
check_values3 = list(post_proc3.values())
f.close()

result = {}
for wrd, replace_list in replace_wrd:
    replace_list = replace_list.split()
    for i in range(len(replace_list)):
        if replace_list[i][0] == '_': replace_list[i] = '(?: |^)' + replace_list[i][1:]
        replace_list[i] = replace_list[i].replace('_', ' ')
    replace_list = '|'.join(replace_list)
    result[wrd] = replace_list

replace_wrd = result
del result

f = open(os.path.join(main_dir, 'dictionary', 'user.dic'), mode='r', encoding='utf-8')
user_word = f.read().strip().split('\n')
user_word = [attr.split('\t')[0] for attr in user_word if len(attr.split('\t')[0]) > 1 and 'NN' in attr.split('\t')[1]]
user_word.sort()

komoran = Komoran("STABLE")
komoran.set_user_dic(os.path.join(main_dir, 'dictionary', 'user.dic'))
komoran.set_fw_dic(os.path.join(main_dir, 'dictionary', 'fwd.dic'))

spacer = ChatSpace()


def preproc_ko_basic(comment):
    comment = hanja.translate(comment, 'substitution')  # 大韓民國은 --> 대한민국은
    comment = comment.replace('ʼ', "'").upper()
    comment = re.sub(r'\s', ' ', comment)
    comment = re.sub(r'[\~\-]+', '', comment)
    comment = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣A-Z0-9\'\:\,\!\?\.\s\(\)]', ' ', comment).strip()
    comment = re.sub(r'\s[ㄱ-ㅎㅏ-ㅡ](\s|,)', ' ', comment)
    comment = re.sub(r'(\d+\:\d+)([가-힣A-Z])', r'\1 \2', comment)
    comment = re.sub(r'[ㄱㅋㄲㄷㅌㅎ그크킄큐키킥킼하핳흐흫히힣]{2,}', ' ', comment)
    comment = re.sub(r'(?<=[ㄱ-ㅎ][ㅏ-ㅣ])[ㅏ-ㅣ]+', ' ', comment)
    comment = re.sub(r'(?<=[가-힣])[ㅏ-ㅣ]+', ' ', comment)
    comment = re.sub(r'[ㅏ-ㅣ]{2,}', ' ', comment)
    comment = re.sub(r'[퓨뮤ㅍㅠㅜ\.]{2,}', ' ', comment)
    comment = re.sub(r'[아악앜앟]+', '아', comment)
    comment = re.sub(r'[어억엌엏]+', '어', comment)
    comment = re.sub(r'[워웡웤]+', '워', comment)
    comment = re.sub(r'[\!\?\.\,]+', ' ', comment)
    comment = re.sub(r'넘\s', r' 너무 ', comment)
    comment = re.sub(r'절대', r' 절대 ', comment)
    comment = re.sub(r'\s+', ' ', comment)
    return comment.strip()


def preproc_ko_word(comment, replace_wrd):
    for line in list(replace_wrd.items()):
        wrd, rex = line[0], re.compile(r'{}'.format('(' + line[1] + ')'))
        if bool(rex.findall(comment)): comment = re.sub(rex, r' {}'.format(wrd), comment)
    return comment.strip()


def preproc_ko_eomi(comment, replace_eomi):
    for line in list(replace_eomi.items()):
        wrd, rex = line[0], re.compile(r'([가-힣])({})'.format(line[1]))
        if bool(rex.findall(comment)): comment = re.sub(rex, r'\1{}'.format(wrd[1:]), comment)
    return comment.strip()


def preproc_ko(comment, replace_wrd, replace_eomi):
    comment = comment + ' '
    comment = preproc_ko_word(comment, replace_wrd)
    comment = preproc_ko_eomi(comment, replace_eomi)
    comment = re.sub(r'^[ㅋㅎ]+', ' ', comment).strip()
    comment = re.sub(r'([ㄱ-ㅎ]) ([ㅏ-ㅣ])', r'\1\2', comment)
    comment = re.sub(r'([ㅏ-ㅣ])([가-힣])', r'\1 \2', comment)
    comment = re.sub(r'([가-힣])([ㅇㅋㅎㅏ-ㅣ])', r'\1 \2', comment)
    comment = re.sub(r'(귀엽|아쉽|예쁘|좋|즐겁)(?: |$)', r'\1다 ', comment)
    comment = re.sub(r'\s+', ' ', comment).strip()
    return comment.strip()


def select_tag_ko(comment):
    rex = re.compile(r'(\d+)\/SN\t\:\/SP\t(\d+)\/SN')  # 동영상 시간 Token -> 고유 명사 변환
    if bool(rex.search(comment)): comment = re.sub(rex, r'\1:\2/NNP', comment)

    rex = re.compile(r'(\d+)\/SN\t\.\/SF\t(\d+)\/SN')  # 소수점 포함 -> 숫자(SN) Tag 변환
    if bool(rex.search(comment)): comment = re.sub(rex, r'\1.\2/SN', comment)

    rex = re.compile(r'(\d+)\/SN\t([^\t]+)(\/N+)')
    if bool(rex.search(comment)): comment = re.sub(rex, r'\1 \2\3', comment)

    rex = re.compile(r'\t(\S)\/NNG\t(겹\/VA)')
    if bool(rex.search(comment)): comment = re.sub(rex, r'\t\1\2', comment)

    rex = re.compile(r'([^\t]+)\/(?:NNG|XR)+\t([^\t]+)\/XSA')
    if bool(rex.search(comment)): comment = re.sub(rex, r'\1\2/VA', comment)

    rex = re.compile(r'([^\t제저])\/XPN+\t([^\t])\/(?:NNG|NNP)')
    if bool(rex.search(comment)): comment = re.sub(rex, r'\1\2/NNG', comment)

    select_tags = ['NNG', 'NNP', 'XPN', 'XSA', 'VA', 'XR', 'SL', 'SJ', 'SN']
    result = re.compile(r'([^\t]+)(?:{})'.format('\/' + '|\/'.join(select_tags))).findall(comment)

    unknown_list = re.compile(r'([^\t]+)(?:\/NA)').findall(comment)

    return ('\t'.join(result), '\t'.join(unknown_list))


ko_data = Parallel(n_jobs=num_cores)(delayed(preproc_ko_basic)(comment) for comment in ko_data)
ko_data = Parallel(n_jobs=num_cores)(delayed(preproc_ko)(comment, replace_wrd, replace_eomi) for comment in ko_data)
ko_data = spacer.space(ko_data, batch_size=64, custom_vocab=user_word)
raw_ko_data = ko_data

ko_data = ['\t'.join(map(str, komoran.get_list(comment))) for comment in ko_data]
ko_data = Parallel(n_jobs=num_cores)(delayed(select_tag_ko)(comment) for comment in ko_data)
ko_data = np.array(ko_data)

raw = []
unknown = []
for i,_ in enumerate(range(len(ko_data))):
    if list(ko_data[:,1][i]):
        raw.append(raw_ko_data[i])
        unknown.append(list(ko_data[:,1][i]))
unknown_inside = {'Unknowns':unknown, 'Raw_comment': raw}
unknown_list = pd.DataFrame(unknown_inside)

del raw
del unknown

ko_data = ko_data[:, 0].tolist()

def build_word_graph(article, gram):
    counter = CountVectorizer(token_pattern=r'[^\t]+', ngram_range=gram)
    word_matrix = normalize(counter.fit_transform(article).toarray().astype(float),axis=0)  # (row, word) weight => CSR-Sparse Matrix
    word_matrix = np.dot(word_matrix.T, word_matrix)

    vocab = counter.vocabulary_
    vocab = dict([(idx, token.upper()) for idx, token in zip(vocab.values(), vocab.keys())])
    return word_matrix, vocab


def get_ranks(graph, d):  # d = damping factor
    A = graph
    matrix_size = graph.shape[0]
    for id in range(matrix_size):
        A[id, id] = 0
        link_sum = np.sum(A[:, id])
        if link_sum != 0: A[:, id] /= link_sum
        A[:, id] *= -d
        A[id, id] = 1
        B = (1 - d) * np.ones((matrix_size, 1))
    ranks = np.linalg.solve(A, B)  # 연립방정식 Ax = b
    return {idx: r[0] for idx, r in enumerate(ranks)}


def keywords_extract(input_data, d, word_num, gram=(1, 2)):
    word_graph, vocab = build_word_graph(input_data, gram)
    rank_idx = get_ranks(word_graph, d)
    keywords = [(vocab[idx], round(wght, 5)) for idx, wght in rank_idx.items()]
    keywords = DataFrame(keywords, columns=['keyword', 'weight']).sort_values(by='weight', ascending=False)
    #     keywords['weight'] = [round(float(v)/sum(keywords['weight']), 5) for v in keywords['weight']]
    keywords = keywords.reset_index(drop=True).iloc[:word_num, :]
    #     keywords = dict(zip(keywords.noun, keywords.weight))

    for i, _ in enumerate(range(len(keywords))):
        check = keywords['keyword'][i].split(' ')
        c = len(check)
        if c == 3:
            if check[0] == check[1] and check[1] == check[2]:
                keywords['keyword'] = check[0]
                c = 1
            elif check[0] == check[1]:
                keywords['keyword'][i] = check[0] + ' ' + check[2]
                check[1] = check[2]
                print('removed check[0]' + check[0])
                del check[2]
                c = 2
            elif check[1] == check[2]:
                keywords['keyword'][i] = check[0] + ' ' + check[1]
                print('removed check[1]' + check[1])
                del check[2]
                c = 2
            elif check[0] == check[2]:
                keywords['keyword'][i] = check[0] + ' ' + check[1]
                print('removed check[2]' + check[2])
                del check[2]
                c = 2
            else:
                if check[0] in check_values1:
                    check[0] = check_key1[check_values1.index(check[0])]
                if check[1] in check_values1:
                    check[1] = check_key1[check_values1.index(check[1])]
                if check[2] in check_values2:
                    check[2] = check_key2[check_values2.index(check[2])]
                keywords['keyword'][i] = check[0] + ' ' + check[1] + ' ' + check[2]
        if c == 2:
            if check[0] == check[1]:
                keywords['keyword'][i] = check[0]
                if keywords['keyword'][i] in check_values2:
                    keywords['keyword'][i] = check_key2[check_values2.index(check[0])]
            else:
                if check[0] in check_values1:
                    check[0] = check_key1[check_values1.index(check[0])]
                if check[1] in check_values2:
                    check[1] = check_key2[check_values2.index(check[1])]
                keywords['keyword'][i] = check[0] + ' ' + check[1]
        if c == 1:
            if keywords['keyword'][i] in check_values2:
                keywords['keyword'][i] = check_key2[check_values2.index(check[0])]

    for i, _ in enumerate(range(len(keywords))):
        check = keywords['keyword'][i].replace(" ", "")
        if check in check_values3:
            keywords['keyword'][i] = check_key3[check_values3.index(check)]

    return keywords



keywords = keywords_extract(ko_data, 0.80, 20, gram=(2,2))
print(list(keywords['keyword']))

keywords = keywords_extract(ko_data, 0.90, 500, gram = (1,2))


text_cloud = []
for i in range(len(keywords)):
    keys = keywords['keyword'][i]
    rep = int(round(keywords['weight'][i]))
    for k,_ in enumerate(range(rep)):
        text_cloud.append(keys)

text_cloud = ' '.join([str(elem) for elem in text_cloud])
pengsu_mask = np.array(Image.open('/home/taejin/Desktop/Project/Keywords_Extraction/image/pengsu_1.png'))

im = Image.open('/home/taejin/Desktop/Project/Keywords_Extraction/image/pengsu_1.png')

wordcloud = WordCloud(width = list(im.size)[0], height = list(im.size)[1], font_path='/home/taejin/Desktop/Project/Keywords_Extraction/font/HumanmoeumT_botong.TTF', mask = pengsu_mask, scale = 5, relative_scaling = 0.6, collocations = False, background_color = "white").generate(text_cloud)
image_colors = ImageColorGenerator(pengsu_mask)
# plt.figure(figsize=[7,7])
plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation = 'bilinear')
plt.axis("off")
plt.tight_layout(pad = 0)
# plt.savefig("/home/taejin/Desktop/Project/Keywords_Extraction/image/test_pengsu.png", format="png")
plt.show()