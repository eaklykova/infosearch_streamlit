import gensim
import compress_fasttext
print('Загрузка модели...')
big_model = gensim.models.fasttext.FastTextKeyedVectors.load('araneum_none_fasttextcbow_300_5_2018/araneum_none_fasttextcbow_300_5_2018.model')
print('Сжатие модели...')
small_model = compress_fasttext.prune_ft_freq(big_model, pq=True)
print('Сохранение модели...')
small_model.save('araneum_none_fasttextcbow_300_5_2018_small.model')
