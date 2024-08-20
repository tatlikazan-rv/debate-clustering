# WARNING
# due to the bigartm/bigartm10 package only being available on linux
# this file/model can only be trained/run on linux devices
import artm
import nltk.corpus
from sklearn.feature_extraction.text import CountVectorizer
from numpy import array

import topic_modeling_utils

cv = CountVectorizer(stop_words=nltk.corpus.stopwords.words('english'))
data_path = "./data/"
nouns = []
with open(data_path + "nouns_only_lines.txt", encoding="utf8") as f:
    for i, line in enumerate(f.read().splitlines()):
        nouns.append(line)

n_wd = array(cv.fit_transform(nouns).todense()).T
vocab = cv.get_feature_names_out()

vectorizer = artm.BatchVectorizer(data_format='bow_n_wd', n_wd=n_wd, vocabulary=vocab)
dictionary = vectorizer.dictionary
model_artm = artm.ARTM(num_topics=10, cache_theta=True, scores=[artm.PerplexityScore(name='PerplexityScore', dictionary=dictionary)], regularizers=[artm.SmoothSparseThetaRegularizer(name='SparseTheta', tau=-0.15)])

model_artm.scores.add(artm.TopTokensScore(name='TopTokensScore', num_tokens=10))

model_artm.scores.add(artm.SparsityPhiScore(name='SparsityPhiScore'))

model_artm.scores.add(artm.TopicKernelScore(name='TopicKernelScore', probability_mass_threshold=0.3))

model_artm.regularizers.add(artm.SmoothSparsePhiRegularizer(name='SparsePhi', tau=-0.1))

model_artm.regularizers.add(artm.DecorrelatorPhiRegularizer(name='DecorrelatorPhi', tau=1.5e+5))

model_artm.num_document_passes = 1

model_artm.initialize(dictionary=dictionary)
model_artm.fit_offline(batch_vectorizer=vectorizer, num_collection_passes=1000)

l = []
for topic_name in model_artm.topic_names:
    l.append(model_artm.score_tracker['TopTokensScore'].last_tokens[topic_name])

topic_modeling_utils.plot_top_words_table(l, 10, "BigARTM")