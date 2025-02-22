##
import re
import json
import base64
import torch
import pickle
import spacy_udpipe
import numpy as np
import streamlit as st
from scipy import sparse
from nltk.corpus import stopwords
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize
from transformers import AutoTokenizer, AutoModel


##
@st.cache(allow_output_mutation=True)
def get_texts_from_corpus(filename):
    """
    Считывает предобработанный корпус из файла.
    :param filename: имя файла с предобработанным корпусом
    :return: корпус или None при ошибке
    """
    with open(filename, 'r', encoding='utf-8') as f:
        texts = json.load(f)['texts']
    return texts


##
@st.cache(allow_output_mutation=True)
def get_stopwords(language='russian'):
    blacklist = stopwords.words(language)
    return blacklist


##
@st.cache(allow_output_mutation=True)
def load_udpipe_model(model_path: str):
    """
    Загружает UDPipe2-модель или сообщает об ошибке.
    :param model_path: имя файла с моделью
    :return: модель или None
    """
    nlp = spacy_udpipe.load_from_path(lang='ru', path=model_path)
    return nlp


##
def parse_with_udpipe(nlp, blacklist, text):
    """
    Предобрабатывает один текст: очищает от пунктуации, приводит к нижнему
    регистру, лемматизирует и убирает стоп-слова.
    :param nlp: udpipe2-модель для лемматизации
    :param blacklist: стоп-слова
    :param text: текст на предобработку
    :return: предобработанный текст
    """
    # очистка от пунктуации и мусора, приведение к нижнему регистру
    # сам udpipe токенизирует очень плохо
    no_punct = re.sub(r'[^\w\s]', ' ', text.lower())

    # лемматизация и очистка от стоп-слов
    doc = nlp(no_punct)
    lemmas = [token.lemma_ for token in doc if token.lemma_ not in blacklist]

    return ' '.join(lemmas)


##
@st.cache(allow_output_mutation=True)
def load_fasttext_model(model_path: str):
    """
    Загружает fasttext-модель для векторизации текстов.
    :param model_path: имя файла с fasttext-моделью
    :return: модель или None
    """
    model = KeyedVectors.load(model_path)
    return model


##
def fasttext_one_doc(text, model):
    """
    Векторизует один текст с использованием Fasttext.
    :param text: предобработанный текст
    :param model: fasttext-модель
    :return: вектор текста
    """
    lemmas = text.split()
    lem_vectors = np.zeros((len(lemmas), model.vector_size))

    # берем вектор слова, если оно есть в модели
    for idx, lemma in enumerate(lemmas):
        if lemma in model:
            lem_vectors[idx] = model[lemma]

    # получаем вектор документа как среднее векторов слов
    if lem_vectors.shape[0] != 0:
        one_text_vec = np.mean(lem_vectors, axis=0)
        # нормализуем
        one_text_vec = one_text_vec / np.linalg.norm(one_text_vec)
    else:  # или вектор нулей, если текст пустой
        one_text_vec = np.zeros((model.vector_size,))

    return one_text_vec


##
def cls_pooling(model_output):
    """
    Получает вектор текста из выдачи Bert.
    :param model_output: выдача Bert по одному документу
    :return: вектор документа
    """
    return model_output[0][:, 0]


##
@st.cache(allow_output_mutation=True)
def load_bert_model(model_name: str):
    """
    Загружает Bert-модель.
    :param model_name: имя модели
    :return: токенизатор и модель
    """
    bert_tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert_model = AutoModel.from_pretrained(model_name)
    return bert_tokenizer, bert_model


##
def vectorize_query_tf_or_count(nlp, blacklist, query, vectorizer):
    """
    Векторизует запрос с помощью Count- или TfIdf-Vectorizer.
    :param nlp: udpipe2-модель для предобработки запроса
    :param blacklist: стоп-слова
    :param query: запрос
    :param vectorizer: инстанс Count- или TfIdf-Vectorizer
    :return: нормализованный вектор запроса
    """
    clean_query = parse_with_udpipe(nlp, blacklist, query)
    q_vect = vectorizer.transform([clean_query])
    # нормализуем
    return normalize(q_vect)


##
def vectorize_query_bm25(nlp, blacklist, query, vectorizer):
    """
    Векторизует запрос с помощью CountVectorizer для BM25.
    :param nlp: udpipe2-модель для предобработки запроса
    :param blacklist: стоп-слова
    :param query: запрос
    :param vectorizer: инстанс CountVectorizer
    :return: вектор запроса
    """
    clean_query = parse_with_udpipe(nlp, blacklist, query)
    q_vect = vectorizer.transform([clean_query])
    # не нормализуем
    return q_vect


##
def vectorize_query_fasttext(nlp, blacklist, query, model):
    """
    Векторизует запрос с помощью Fasttext.
    :param nlp: udpipe2-модель для предобработки запроса
    :param blacklist: стоп-слова
    :param query: запрос
    :param model: Fasttext-модель
    :return: нормализованный вектор запроса
    """
    clean_query = parse_with_udpipe(nlp, blacklist, query)
    q_vect = fasttext_one_doc(clean_query, model)
    # здесь возвращается нормализованный вектор
    return q_vect


##
def vectorize_query_bert(query, bert_model):
    """
    Векторизует запрос с помощью Bert.
    :param query: запрос
    :param bert_model: Bert-модель
    :return: нормализованный запрос
    """
    tokenizer, model = bert_model
    encoded_input = tokenizer(query, padding=True, truncation=True,
                              max_length=512, return_tensors='pt')
    with torch.no_grad():
        output = model(**encoded_input)
    q_vect = cls_pooling(output)
    # нормализуем
    return normalize(q_vect)


##
def get_similarity(corpus_matr, q_vect):
    """
    Вычисляет близость запроса с документами корпуса.
    :param corpus_matr: векторизованный корпус
    :param q_vect: векторизованный запрос
    :return: вектор; i-й элемент = близость запроса с i-м документом корпуса
    """
    sim_vect = corpus_matr.dot(q_vect.T)
    # дальше сортировка через аргсорт, а она не работает со спарс-матрицами
    if sparse.issparse(sim_vect):
        sim_vect = sim_vect.toarray()
    return np.array(sim_vect)


##
def sort_files_by_similarity(sim_vect, texts):
    """
    Сортирует документы по убыванию релевантности.
    :param sim_vect: вектор близости запроса с документами корпуса
    :param texts: тексты корпуса
    :return: матрица текстов и близостей запроса с ними
    """
    # сортируем вектор близости по убыванию
    sorted_idx = np.argsort(sim_vect, axis=0)[::-1]

    # сортируем тексты и близости через маску
    sorted_files = np.array(texts)[sorted_idx.ravel()]
    sorted_scores = sim_vect[sorted_idx.ravel()]

    # то же, что zip, но специально для np.array
    # задаем тип sorted_files, иначе sorted_scores переводятся в строку
    files_with_scores = np.column_stack((np.array(sorted_files, dtype=object),
                                         sorted_scores))
    return files_with_scores


##
@st.cache(allow_output_mutation=True)
def get_countvectorizer_components():
    """
    Загружает компоненты для поиска с CountVectorizer.
    :return: матрица ответов и векторайзер или None
    """
    ans_matr = sparse.load_npz('files/cv_answers.npz')
    ques_matr = sparse.load_npz('files/cv_questions.npz')
    vect_or_model = pickle.load(open('files/cv_vect.pickle', 'rb'))
    return ans_matr, ques_matr, vect_or_model


##
@st.cache(allow_output_mutation=True)
def get_tfidfvectorizer_components():
    """
    Загружает компоненты для поиска с TfIdfVectorizer.
    :return: матрица ответов и векторайзер или None
    """
    ans_matr = sparse.load_npz('files/tfidf_answers.npz')
    ques_matr = sparse.load_npz('files/tfidf_questions.npz')
    vect_or_model = pickle.load(open('files/tfidf_vect.pickle', 'rb'))
    return ans_matr, ques_matr, vect_or_model


##
@st.cache(allow_output_mutation=True)
def get_bm25_components():
    """
    Загружает компоненты для поиска с Okapi BM25.
    :return: матрица ответов и векторайзер или None
    """
    ans_matr = sparse.load_npz('files/bm25_answers.npz')
    ques_matr = sparse.load_npz('files/bm25_questions.npz')
    vect_or_model = pickle.load(open('files/bm25_vect.pickle', 'rb'))
    return ans_matr, ques_matr, vect_or_model


##
@st.cache(allow_output_mutation=True)
def get_fasttext_components():
    """
    Загружает компоненты для поиска с Fasttext.
    :return: матрица ответов и fasttext-модель или None
    """
    ans_matr = np.load('files/fasttext_answers.npy')
    ques_matr = np.load('files/fasttext_questions.npy')
    vect_or_model = load_fasttext_model('''araneum_none_fasttextcbow_300_5_\
2018/araneum_none_fasttextcbow_300_5_2018.model''')
    return ans_matr, ques_matr, vect_or_model


##
@st.cache(allow_output_mutation=True)
def get_bert_components():
    """
    Загружает компоненты для поиска с Bert.
    :return: матрица ответов и bert-модель или None
    """
    ans_matr = torch.load('files/bert_answers.pt')
    ques_matr = torch.load('files/bert_questions.pt')
    vect_or_model = load_bert_model('cointegrated/rubert-tiny')
    return ans_matr, ques_matr, vect_or_model


##
@st.cache(suppress_st_warning=True, allow_output_mutation=True)
def load_components():
    """
    Загружает все компоненты, необходимые для поиска.
    :return: необходимые компоненты или None при ошибке на любом из этапов
    """
    # тексты, которые будут печататься пользователю
    output = get_texts_from_corpus('texts.json')
    blacklist = get_stopwords()
    nlp = load_udpipe_model('russian-taiga-ud-2.5-191206.udpipe')

    cv_a_matr, cv_q_matr, cv_vect = get_countvectorizer_components()
    tf_a_matr, tf_q_matr, tf_vect = get_tfidfvectorizer_components()
    bm_a_matr, bm_q_matr, bm_vect = get_bm25_components()
    ft_a_matr, ft_q_matr, ft_vect = get_fasttext_components()
    bt_a_matr, bt_q_matr, bt_vect = get_bert_components()

    return output, blacklist, nlp, cv_a_matr, cv_q_matr, cv_vect, tf_a_matr,\
        tf_q_matr, tf_vect, bm_a_matr, bm_q_matr, bm_vect, ft_a_matr,\
        ft_q_matr, ft_vect, bt_a_matr, bt_q_matr, bt_vect


##
def search_with_countvectorizer(query, output, blacklist, nlp, cv_matr,
                                cv_vect):
    q_vect = vectorize_query_tf_or_count(nlp, blacklist, query, cv_vect)
    sim_vect = get_similarity(cv_matr, q_vect)
    files_with_scores = sort_files_by_similarity(sim_vect, output)
    return files_with_scores


##
def search_with_tfidfvectorizer(query, output, blacklist, nlp, tf_matr,
                                tf_vect):
    q_vect = vectorize_query_tf_or_count(nlp, blacklist, query, tf_vect)
    sim_vect = get_similarity(tf_matr, q_vect)
    files_with_scores = sort_files_by_similarity(sim_vect, output)
    return files_with_scores


##
def search_with_bm25(query, output, blacklist, nlp, bm_matr, bm_vect):
    q_vect = vectorize_query_bm25(nlp, blacklist, query, bm_vect)
    sim_vect = get_similarity(bm_matr, q_vect)
    files_with_scores = sort_files_by_similarity(sim_vect, output)
    return files_with_scores


##
def search_with_fasttext(query, output, blacklist, nlp, ft_matr, ft_vect):
    q_vect = vectorize_query_fasttext(nlp, blacklist, query, ft_vect)
    sim_vect = get_similarity(ft_matr, q_vect)
    files_with_scores = sort_files_by_similarity(sim_vect, output)
    return files_with_scores


##
def search_with_bert(query, output, bt_matr, bt_vect):
    q_vect = vectorize_query_bert(query, bt_vect)
    sim_vect = get_similarity(bt_matr, q_vect)
    files_with_scores = sort_files_by_similarity(sim_vect, output)
    return files_with_scores


##
@st.cache(allow_output_mutation=True)
def set_background(bin_file):
    """
    Функция, обеспечивающая отображение фона.
    :param bin_file: имя файла с изображением
    :return:
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-position: center;
    }
    </style>
    ''' % bin_str
    return page_bg_img


##
if __name__ == '__main__':
    pass
