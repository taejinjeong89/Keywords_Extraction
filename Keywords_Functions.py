#!/usr/bin/python3
import os, warnings, re
warnings.filterwarnings(action="ignore")

import numpy as np
from pandas import DataFrame
from joblib import Parallel, delayed
import fasttext
from chatspace import ChatSpace
from PyKomoran import *
import hanja
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize

## Preprocessing Part

class preprocessor():

    def __init__(self, main_dir):
        self.main_dir = main_dir

    def update(self):
        f = open(os.path.join(self.main_dir, 'dictionary', 'user.dic'), mode='r', encoding='utf-8')
        reorder_user_dic = list(f)

        for i, _ in enumerate(range(len(reorder_user_dic))):
            if not reorder_user_dic[i].endswith('\n'):
                reorder_user_dic[i] = reorder_user_dic[i] + '\n'
        reorder_user_dic = sorted(reorder_user_dic, key=len, reverse=True)
        f.close()

        f = open(os.path.join(self.main_dir, 'dictionary', 'user.dic'), mode='w', encoding='utf-8')
        f.writelines([line for line in reorder_user_dic])
        f.close()
        del reorder_user_dic

        f = open(os.path.join(self.main_dir, 'dictionary', 'replace_eomi.dic'), mode='r', encoding='utf-8')
        replace_eomi = f.read().strip().split('\n')
        replace_eomi = [line.split('\t') for line in replace_eomi if line.strip()[0] != '#']
        replace_eomi = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in replace_eomi])
        f.close()
        self.replace_eomi = replace_eomi

        f = open(os.path.join(self.main_dir, 'dictionary', 'replace_wrd.dic'), mode='r', encoding='utf-8')
        replace_wrd = f.read().strip().split('\n')
        replace_wrd = [line.split('\t') for line in replace_wrd if line.strip()[0] != '#']
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
        self.replace_wrd = replace_wrd

        f = open(os.path.join(self.main_dir, 'dictionary', 'user.dic'), mode='r', encoding='utf-8')
        user_word = f.read().strip().split('\n')
        user_word = [attr.split('\t')[0] for attr in user_word if
                     len(attr.split('\t')[0]) > 1 and 'NN' in attr.split('\t')[1]]
        user_word.sort()
        f.close()
        self.user_word = user_word

        komoran = Komoran("STABLE")
        komoran.set_user_dic(os.path.join(self.main_dir, 'dictionary', 'user.dic'))
        komoran.set_fw_dic(os.path.join(self.main_dir, 'dictionary', 'fwd.dic'))
        self.komoran = komoran

    def spacer(self, comment, batch_size = 1):
        spacer = ChatSpace()
        comment = spacer.space(comment, batch_size=batch_size, custom_vocab=self.user_word)
        return comment

    def tag_comment(self, comment, num_cores = 1):
        data = ['\t'.join(map(str, self.komoran.get_list(line))) for line in comment]
        data = Parallel(n_jobs=num_cores)(delayed(self.select_tag_ko)(line) for line in data)
        data = np.array(data)
        return data

    def lang_classify(self, comment):
        lang_gbn_model = os.path.join(self.main_dir, 'dictionary', 'lid.176.bin')
        lang_gbn_model = fasttext.load_model(lang_gbn_model)

        # 개행 문자 -> 띄어쓰기 변경
        comment = comment.replace('\n', ' ').strip()
        # 정규식 기반 필터
        if bool(re.match(r'^[^a-zA-Z0-9가-힣]+$', comment)): result = (comment, 'unknown', 0)
        # 5자 이상 and 2개 문자 이상 사용
        elif bool(len(comment) >= 5 and len(set(comment)) > 1) == False: result = (comment, 'unknown', 0)
        else:
            result = lang_gbn_model.predict(comment)
            result = (comment, result[0][0].replace('__label__', '').strip(), round(result[1][0], 5))
        return list(result)

    def ko_basic(self, comment):
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

    def ko_word(self, comment):
        for line in list(self.replace_wrd.items()):
            wrd, rex = line[0], re.compile(r'{}'.format('(' + line[1] + ')'))
            if bool(rex.findall(comment)): comment = re.sub(rex, r' {}'.format(wrd), comment)
        return comment.strip()

    def ko_eomi(self, comment):
        for line in list(self.replace_eomi.items()):
            wrd, rex = line[0], re.compile(r'([가-힣])({})'.format(line[1]))
            if bool(rex.findall(comment)): comment = re.sub(rex, r'\1{}'.format(wrd[1:]), comment)
        return comment.strip()

    def ko_more(self, comment):
        comment = comment + ' '
        comment = self.ko_word(comment)
        comment = self.ko_eomi(comment)
        comment = re.sub(r'^[ㅋㅎ]+', ' ', comment).strip()
        comment = re.sub(r'([ㄱ-ㅎ]) ([ㅏ-ㅣ])', r'\1\2', comment)
        comment = re.sub(r'([ㅏ-ㅣ])([가-힣])', r'\1 \2', comment)
        comment = re.sub(r'([가-힣])([ㅇㅋㅎㅏ-ㅣ])', r'\1 \2', comment)
        comment = re.sub(r'(귀엽|아쉽|예쁘|좋|즐겁)(?: |$)', r'\1다 ', comment)
        comment = re.sub(r'\s+', ' ', comment).strip()
        return comment.strip()

    def select_tag_ko(self, comment):
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

## Keywords Extraction Part

class keywords_extraction():

    def __init__(self,main_dir):
        self.main_dir = main_dir

    def update(self):
        f = open(os.path.join(self.main_dir, 'dictionary', 'post_processor_1.dic'), mode='r', encoding='utf-8')
        post_proc1 = f.read().strip().split('\n')
        post_proc1 = [line.split('\t') for line in post_proc1 if line.strip()[0] != '#']
        post_proc1 = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in post_proc1])
        check_key1 = list(post_proc1.keys())
        check_values1 = list(post_proc1.values())
        f.close()
        self.check_key1 = check_key1
        self.check_values1 = check_values1

        f = open(os.path.join(self.main_dir, 'dictionary', 'post_processor_2.dic'), mode='r', encoding='utf-8')
        post_proc2 = f.read().strip().split('\n')
        post_proc2 = [line.split('\t') for line in post_proc2 if line.strip()[0] != '#']
        post_proc2 = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in post_proc2])
        check_key2 = list(post_proc2.keys())
        check_values2 = list(post_proc2.values())
        f.close()
        self.check_key2 = check_key2
        self.check_values2 = check_values2

        f = open(os.path.join(self.main_dir, 'dictionary', 'post_processor_3.dic'), mode='r', encoding='utf-8')
        post_proc3 = f.read().strip().split('\n')
        post_proc3 = [line.split('\t') for line in post_proc3 if line.strip()[0] != '#']
        post_proc3 = dict([(line[0], '|'.join(line[1].split(' ')).replace('_', ' ')) for line in post_proc3])
        check_key3 = list(post_proc3.keys())
        check_values3 = list(post_proc3.values())
        f.close()
        self.check_key3 = check_key3
        self.check_values3 = check_values3

    def build_word_graph(self, input_data, gram):
        counter = CountVectorizer(token_pattern=r'[^\t]+', ngram_range=gram)
        # (row, word) weight => CSR-Sparse Matrix
        word_matrix = normalize(counter.fit_transform(input_data).toarray().astype(float),axis=0)
        word_matrix = np.dot(word_matrix.T, word_matrix)
        vocab = counter.vocabulary_
        vocab = dict([(idx, token.upper()) for idx, token in zip(vocab.values(), vocab.keys())])
        return word_matrix, vocab

    def get_ranks(self, graph, d):  # d = damping factor
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

    def keywords_extract(self, input_data, word_num, gram = (2,2), d = 0.80):
        word_graph, vocab = self.build_word_graph(input_data, gram)
        rank_idx = self.get_ranks(word_graph, d)
        keywords = [(vocab[idx], round(wght, 5)) for idx, wght in rank_idx.items()]
        keywords = DataFrame(keywords, columns=['keyword', 'weight']).sort_values(by='weight', ascending=False)
        # keywords['weight'] = [round(float(v)/sum(keywords['weight']), 5) for v in keywords['weight']]
        keywords = keywords.reset_index(drop=True).iloc[:word_num, :]
        # keywords = dict(zip(keywords.noun, keywords.weight))

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
                    if check[0] in self.check_values1:
                        check[0] = self.check_key1[self.check_values1.index(check[0])]
                    if check[1] in self.check_values1:
                        check[1] = self.check_key1[self.check_values1.index(check[1])]
                    if check[2] in self.check_values2:
                        check[2] = self.check_key2[self.check_values2.index(check[2])]
                    keywords['keyword'][i] = check[0] + ' ' + check[1] + ' ' + check[2]
            if c == 2:
                if check[0] == check[1]:
                    keywords['keyword'][i] = check[0]
                    if keywords['keyword'][i] in self.check_values2:
                        keywords['keyword'][i] = self.check_key2[self.check_values2.index(check[0])]
                else:
                    if check[0] in self.check_values1:
                        check[0] = self.check_key1[self.check_values1.index(check[0])]
                    if check[1] in self.check_values2:
                        check[1] = self.check_key2[self.check_values2.index(check[1])]
                    keywords['keyword'][i] = check[0] + ' ' + check[1]
            if c == 1:
                if keywords['keyword'][i] in self.check_values2:
                    keywords['keyword'][i] = self.check_key2[self.check_values2.index(check[0])]

        for i, _ in enumerate(range(len(keywords))):
            check = keywords['keyword'][i].replace(" ", "")
            if check in self.check_values3:
                keywords['keyword'][i] = self.check_key3[self.check_values3.index(check)]

        return keywords