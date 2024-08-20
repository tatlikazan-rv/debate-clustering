import os
from collections import Counter
from typing import List
import tomotopy as tp

import matplotlib
import nltk
from sklearn import metrics
from sklearn.cluster import KMeans
from tomotopy.utils import Corpus
import numpy as np
import utils

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
FILE_NAME = "data/Game_of_Thrones_Script.csv"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(CURRENT_DIR, FILE_NAME)


def tomotopy_corpus() -> Corpus:
    """
    returns tomotopy corpus of the GoT Data
    :return:
    """
    sentences = utils.data_lines()
    corpus = tp.utils.Corpus(stopwords=lambda x: len(x) <= 2 or x in stopwords.words('english'),
                             tokenizer=tp.utils.SimpleTokenizer())
    corpus.process(sentences)
    return corpus


def bow_vectorizer(sen: List[str]):
    """
    Takes a list of sentences and returns a bag of words matrix
    :param sen:
    :return:
    feature_names: list[str] A list of feature names.
    mat: array of shape (n_samples, n_features) Document-term matrix.
    """
    bow_v = CountVectorizer(stop_words=stopwords.words('english'), lowercase=True, max_df=0.5, min_df=5)
    mat = bow_v.fit_transform(sen)
    feature_names = bow_v.get_feature_names()
    return feature_names, mat


def tf_idf_vectorizer(sen: List[str]):
    """
    vectorizes list of strings using TF_IDF
    :param sen:
    :return:
    feature_names: list[str] A list of feature names.
    mat: sparse matrix of (n_samples, n_features) Tf-idf-weighted document-term matrix.
    """
    tfidf_v = TfidfVectorizer(stop_words=stopwords.words('english'), ngram_range=(1, 1), max_df=.8, min_df=.01)
    mat = tfidf_v.fit_transform(sen)
    feature_names = tfidf_v.get_feature_names()
    return feature_names, mat


def plot_top_words_table_sklearn(model, feature_names: List[str], n_top_words: int, title: str = "Title"):
    """
    calls plot_top_words_table for sklearn model
    :param model: sklearn.decomposition model
    :param feature_names: list[str] A list of feature names.
    :param n_top_words:
    :param title:
    :return:
    """
    top_features = []
    for topic in model.components_:
        top_features_ind = topic.argsort()[: -n_top_words - 1: -1]
        top_features.append([feature_names[i] for i in top_features_ind])
    plot_top_words_table(top_features, n_top_words, title)


def plot_top_words_table(top_words_per_topic: List[List[str]], n_top_words: int, title: str = "Title"):
    """
    plots topic-word matrix as multiple tables
    :param top_words_per_topic: List of arrays containing the top words in each topic
    :param n_top_words:
    :param title:
    :return:
    """
    fig, axes = plt.subplots(2, int(n_top_words / 2), figsize=(15, 8), sharex=True)
    axes = axes.flatten()
    for topic_idx, top_words in enumerate(top_words_per_topic):
        ax = axes[topic_idx]
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=[[top_words[i], top_words[i + 5]] for i in range(int(n_top_words / 2))],
                         rowLabels=["{:X}".format(i) for i in range(int(n_top_words / 2))],
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.5, 2)
        ax.set_title(f"Topic {topic_idx + 1}", fontdict={"fontsize": 24})
        fig.suptitle(title, fontsize=30)

    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()


def plot_cluster_metrics_sc(data, min_clusters: int = 3, max_clusters: int = 30, iters: int = 3):
    """
    plots the silhouette coefficient of the given data a range number of clusters
    :param data:
    :param min_clusters:
    :param max_clusters:
    :param iters:
    :return:
    """
    n_clusters = []

    wgss_mean = []  # within group sum of squares, or inertia
    wgss_var = []  # variance of results because it's stochastic

    silhouette_mean = []
    silhouette_var = []

    for n in range(min_clusters, max_clusters + 1):
        n_clusters.append(n)

        wgss_iters = []
        silhouette_iters = []

        for i in range(iters):
            clusterer = KMeans(n_clusters=n).fit(data)
            wgss_iters.append(clusterer.inertia_)
            silhouette_iters.append(metrics.silhouette_score(data,
                                                             clusterer.labels_,
                                                             metric='euclidean'))

        wgss_mean.append(np.array(wgss_iters).mean())
        wgss_var.append(np.array(wgss_iters).var())

        silhouette_mean.append(np.array(silhouette_iters).mean())
        silhouette_var.append(np.array(silhouette_iters).var())

    wgss_mean = np.array(wgss_mean)
    wgss_var = np.array(wgss_var)

    silhouette_mean = np.array(silhouette_mean)
    silhouette_var = np.array(silhouette_var)

    # plot every metric
    plt.figure(figsize=(10, 10))

    plt.subplot(211)
    plt.xticks(n_clusters)
    plt.plot(n_clusters, silhouette_mean, alpha=1, color='red', label='silhouette')
    plt.fill_between(n_clusters,
                     silhouette_mean - silhouette_var,
                     silhouette_mean + silhouette_var,
                     alpha=0.2)
    plt.title("Evaluate quality of clusters")
    plt.grid(True)
    plt.legend()

    plt.subplot(212)
    plt.xticks(n_clusters)
    plt.plot(n_clusters, wgss_mean, alpha=1, label='inertia', color='orange')
    plt.fill_between(n_clusters, wgss_mean - wgss_var, wgss_mean + wgss_var, alpha=0.2)
    plt.xlabel("# clusters")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_cluster_metrics_dbi(data, min_clusters: int = 3, max_clusters: int = 15, iters: int = 3):
    """
    plots the DBI of the given data a range number of clusters
    :param data:
    :param min_clusters:
    :param max_clusters:
    :param iters:
    :return:
    """
    n_clusters = []

    wgss_mean = []  # within group sum of squares, or inertia
    wgss_var = []  # variance of results because it's stochastic

    dbi_mean = []
    dbi_var = []

    for n in range(min_clusters, max_clusters + 1):
        n_clusters.append(n)

        wgss_iters = []
        dbi_iters = []

        for i in range(iters):
            clusterer = KMeans(n_clusters=n).fit(data)
            wgss_iters.append(clusterer.inertia_)
            dbi_iters.append(metrics.davies_bouldin_score(data, clusterer.labels_, ))

        wgss_mean.append(np.array(wgss_iters).mean())
        wgss_var.append(np.array(wgss_iters).var())

        dbi_mean.append(np.array(dbi_iters).mean())
        dbi_var.append(np.array(dbi_iters).var())

    wgss_mean = np.array(wgss_mean)
    wgss_var = np.array(wgss_var)

    dbi_mean = np.array(dbi_mean)
    dbi_var = np.array(dbi_var)

    # plot every metric
    plt.figure(figsize=(10, 10))

    plt.subplot(211)
    plt.xticks(n_clusters)
    plt.plot(n_clusters, dbi_mean, alpha=1, color='red', label='DBI')
    plt.fill_between(n_clusters,
                     dbi_mean - dbi_var,
                     dbi_mean + dbi_var,
                     alpha=0.2)
    plt.title("Evaluate quality of clusters")
    plt.grid(True)
    plt.legend()

    plt.subplot(212)
    plt.xticks(n_clusters)
    plt.plot(n_clusters, wgss_mean, alpha=1, label='inertia', color='orange')
    plt.fill_between(n_clusters, wgss_mean - wgss_var, wgss_mean + wgss_var, alpha=0.2)
    plt.xlabel("# clusters")
    plt.legend()
    plt.grid(True)
    plt.show()


def prep_df(df: pd.DataFrame, topics: List[int], probs: List[float] = None):
    """
    adds the topics and probabilities columns to the Database
    :param df:
    :param topics:
    :param probs:
    :return:
    """
    df["Topics"] = topics
    if probs is not None:
        df["Probabilities"] = probs


def get_name_topic_df(df: pd.DataFrame, probability_threshold: float = 0) -> pd.DataFrame:
    """
     :param probability_threshold: the threshold at which sentences are counted as relevant
     :param df: dataframe containing the characters and their sentences and each topic and probability per sentence
     :return: a dataframe with each character and the top 3 topics they talk about throughout the entire series
     """
    if probability_threshold != 0:
        name_topic = df.drop(df[df["Probabilities"] < probability_threshold].index \
                             .append(df[df["Topics"] == -1].index)) \
            .groupby('Name').agg({'Topics': list})
    else:
        name_topic = df.drop(df[df["Topics"] == -1].index) \
            .groupby('Name').agg({'Topics': list})
    topic_char_matrix = name_topic["Topics"]
    counter_list = [Counter(topic_list) for topic_list in topic_char_matrix]
    top_topic_char_matrix = [sorted([top_topic[0] for top_topic in counter.most_common(3)]) for counter in counter_list]
    name_topic["Topics"] = top_topic_char_matrix

    name_topic["Topic Count"] = name_topic["Topics"].apply(len)
    name_topic = name_topic.sort_values('Topic Count', ascending=False)
    name_topic.index.name = "Name"
    return name_topic.reset_index()


def get_name_topic_per_season_df(df: pd.DataFrame, probability_threshold: float = 0) -> pd.DataFrame:
    """
     :param probability_threshold: the threshold at which sentences are counted as relevant
     :param df: dataframe containing the characters and their sentences
     :return: a dataframe with each character and the top 3 topics they talk about in each season
     """
    if probability_threshold != 0:
        name_topic = df.drop(df[df["Probabilities"] < probability_threshold].index \
                             .append(df[df["Topics"] == -1].index))
    else:
        name_topic = df.drop(df[df["Topics"] == -1].index)
    topics_per_season = {}
    for index, row in name_topic.iterrows():
        if row["Name"] not in topics_per_season:
            topics_per_season[row["Name"]] = [[], [], [], [], [], [], [], []]
        topics_per_season[row["Name"]][int(row["Season"].replace("Season ", "")) - 1].append(row["Topics"])

    for name, season_topics in topics_per_season.items():
        counter_list = [Counter(topic_list) for topic_list in topics_per_season[name]]
        topics_per_season[name] = [sorted([top_topic[0] for top_topic in counter.most_common(3)]) for counter in
                                   counter_list]
    name_topic_season = pd.DataFrame.from_dict(topics_per_season, orient='index',
                                               columns=["Season 1", "Season 2", "Season 3", "Season 4", "Season 5",
                                                        "Season 6", "Season 7", "Season 8"]).sort_index()
    name_topic_season.index.name = "Name"
    return name_topic_season.reset_index()


def filter_by_characters(characters: List[str], df) -> pd.DataFrame:
    """
    filters a data frame with the column "Name" by character names
    :param characters:
    :param df:
    :return:
    """
    return df.loc[df["Name"].isin(characters)]


def get_sentences_for_char(df: pd.Dataframe, char_name: str, season: str, topics: List[int], probability_threshold: float = 0) -> pd.DataFrame:
    """
    gets all sentences spoken by a specific character in the specified topics
    :param df:
    :param char_name:
    :param season:
    :param topics:
    :param probability_threshold:
    :return:
    """
    sentences_per_topic = {}
    total_sentences = {}
    for topic in topics:
        if probability_threshold > 0:
            sentences_in_topic = df[df["Season"] == "Season " + str(season)][df["Topics"] == topic][
                df["Probabilities"] >= probability_threshold]
        else:
            sentences_in_topic = df[df["Season"] == "Season " + str(season)][df["Topics"] == topic]
        sentences_per_topic[f"Topic {topic}"] = sentences_in_topic[df["Name"] == char_name]["Sentence"].reset_index(
            drop=True)
        total_sentences[f"Topic {topic}"] = str(sentences_in_topic["Sentence"].count())
    sentence_count = df[df["Season"] == "Season " + str(season)][df["Name"] == char_name][df["Topics"] != -1].shape[0]
    non_topical_count = df[df["Season"] == "Season " + str(season)][df["Name"] == char_name][df["Topics"] == -1].shape[0]
    print(f"{char_name} has {sentence_count} topical sentences in Season {season}")
    print(f"{char_name} has {non_topical_count} non-topical sentences in Season {season}")
    if probability_threshold > 0:
        prob_topical_count = df[df["Season"] == "Season " + str(season)][df["Name"] == char_name][df["Topics"] != -1][
            df["Probabilities"] >= probability_threshold].shape[0]
        print(
            f"{char_name} has {prob_topical_count} topical sentences with topic probability of at least {probability_threshold} in Season {season}")
    return pd.DataFrame.from_dict(sentences_per_topic).append(total_sentences, ignore_index=True)


def get_sentences(df: pd.DataFrame, topic: int) -> pd.DataFrame:
    """
    gets the sentences that correspond to a certain topic
    :param df:
    :param topic:
    :return:
    """
    return df[df["Topics"] == topic].groupby('Name').agg({'Sentence': list}).reset_index()


def all_topics_per_season(df: pd.DataFrame, probability_threshold: float = 0) -> pd.DataFrame:
    """
    returns a Dataframe with each season and the list of topics mentioned in that season
    :param df:
    :param probability_threshold:
    :return:
    """
    if probability_threshold != 0:
        season_topic = df.drop(df[df["Probabilities"] < probability_threshold].index \
                               .append(df[df["Topics"] == -1].index)) \
            .groupby('Season').agg({'Topics': list})
    else:
        season_topic = df.drop(df[df["Topics"] == -1].index) \
            .groupby('Season').agg({'Topics': list})
    season_topic["Topics"] = season_topic["Topics"].apply(Counter)

    return pd.DataFrame.from_dict(season_topic.to_dict()["Topics"])
