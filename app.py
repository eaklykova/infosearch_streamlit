##
import time
import streamlit as st
from main_funcs import *


##
def search():
    """
    Функция, взаимодействующая с интерфейсом и выполняющая поиск.
    :return: None
    """
    output, blacklist, nlp, cv_a_matr, cv_q_matr, cv_vect, tf_a_matr, \
        tf_q_matr, tf_vect, bm_a_matr, bm_q_matr, bm_vect, ft_a_matr, \
        ft_q_matr, ft_vect, bt_a_matr, bt_q_matr, bt_vect = load_components()

    st.title('Questions about Love')
    page_bg_img = set_background('heart_background.png')
    st.markdown(page_bg_img, unsafe_allow_html=True)

    method = st.sidebar.selectbox(
        'Как ищем?',
        ('CountVectorizer', 'TfIdfVectorizer', 'Okapi BM25',
         'FastText', 'Bert (rubert-tiny)')
    )
    place = st.sidebar.radio(
        'Где ищем?',
        ('в вопросах', 'в ответах')
    )
    res_num = st.sidebar.slider(
        'Сколько результатов напечатать?',
        5, 20
    )

    st.sidebar.image('heart_ornament.png', use_column_width=True)
    query = st.text_input('Что ищем?')
    st.button('Поехали!')

    start = time.time()

    if query.isspace():
        st.write('Ой! Это пустой запрос :(')

    elif query:

        if place == 'в вопросах':
            cv_matr, tf_matr, bm_matr, ft_matr, bt_matr = cv_q_matr, tf_q_matr,\
                                                          bm_q_matr, ft_q_matr,\
                                                          bt_q_matr
        else:
            cv_matr, tf_matr, bm_matr, ft_matr, bt_matr = cv_a_matr, tf_a_matr,\
                                                          bm_a_matr, ft_a_matr,\
                                                          bt_a_matr

        if method == 'CountVectorizer':
            results = search_with_countvectorizer(query, output, blacklist, nlp,
                                                  cv_matr, cv_vect)

        elif method == 'TfIdfVectorizer':
            results = search_with_tfidfvectorizer(query, output, blacklist, nlp,
                                                  tf_matr, tf_vect)

        elif method == 'Okapi BM25':
            results = search_with_bm25(query, output, blacklist, nlp,
                                       bm_matr, bm_vect)

        elif method == 'FastText':
            results = search_with_fasttext(query, output, blacklist, nlp,
                                           ft_matr, ft_vect)

        elif method == 'Bert (rubert-tiny)':
            results = search_with_bert(query, output, bt_matr, bt_vect)

        top_n = results[:res_num]
        to_print = '\n'.join(['{}. {} (близость с запросом {})'.format(
            i+1, res[0], round(res[1], 3)) for i, res in enumerate(top_n)])

        st.write(to_print)

        end = time.time()
        inference_time = round((end - start) * 1000)
        st.write(f"<p style='opacity:.5'>Поиск занял {inference_time} мс</p>",
                 unsafe_allow_html=True)


##
if __name__ == '__main__':
    search()
