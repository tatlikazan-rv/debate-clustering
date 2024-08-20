import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from scipy.cluster.hierarchy import linkage, fcluster
import seaborn as sns



#----------- Embeddings -----------#

def compute_avg_word_embeddings(line: str, word_embeddings: dict) -> list:
    """
    Retrieves a word embedding for each word of the line, averages them.

    Parameters
    ----------
    line : str
        Line of text
    word_embeddings : dict
        Dictionary with words as keys and embeddings as values

    Returns
    -------
    list
    """
    
    e = []
    for token in line:
        try:
            e.append(word_embeddings[token])
        except:
            pass
        
    if len(e) == 0:
        return None
        
    return np.mean(e, axis=0)
        
    
def get_glove_embeddings() -> dict:
    """
    Reads pre-computed glove embeddings from disk and transforms them into a dictionary 
    with words as keys and embeddings as values.

    Returns
    -------
    dict
    """
    
    glove_path = "embeddings/glove.6B/glove.6B.300d.txt"

    try:
        with open(glove_path, "r", encoding = "utf8") as f:
            glove = [line[:-1] for line in f.readlines()]
    except Exception as e:
        print("The embedding needs to be in the path: embeddings/glove.6B/glove.6B.300d.txt")
        print("Download the embedding from https://nlp.stanford.edu/data/glove.6B.zip.")
        return
        
    glove_dict = {}

    for token in glove:
        embedding = [float(token) for token in token.split(" ")[1:]]
        
        glove_dict[token.split(" ")[0]] = np.array(embedding)
        
    return glove_dict

def get_w2v_embeddings(file_name: str) -> dict:
    """
    Reads pre-computed word2vec model from disk.
    
    Parameters
    ----------
    file_name : str
        Name of the file with word2vec model.
  
    Returns
    -------
    dict
    """
    
    w2v_path = f"argument_clustering/embeddings/word2vec/{file_name}"
    
    try:
        w2v_model = Word2Vec.load(w2v_path)
    except:
        print(f"The model should be located at {w2v_path}.")
        print("Otherwise, you need to train it.")
        return
    
    return w2v_model.wv

def get_d2v_embeddings(file_name: str) -> dict:
    """
    Reads pre-computed doc2vec model from disk.
    
    Parameters
    ----------
    file_name : str
        Name of the file with doc2vec model.
  
    Returns
    -------
    dict
    """
    
    d2v_path = f"argument_clustering/embeddings/doc2vec/{file_name}"
    d2v_model = Doc2Vec.load(d2v_path)
    
    return d2v_model.dv


#----------- Argument Clustering -----------#

def cosine_dist(embeddings: pd.Series()) -> np.ndarray:
    """
    Computes pair-wise cosine distance between embeddings.
    
    Parameters
    ----------
    embeddings : pd.Series()
        Column (list) of embeddings
  
    Returns
    -------
    np.ndarray
    """
    
    embeddings = list(embeddings)
    dist = 1-cosine_similarity(embeddings)
    
    return dist

def get_linkage(dist, method='complete') -> np.ndarray:
    """
    Computes linkage from a distance matrix using the specified method. 
    
    For available methods and their descriptions check scipy documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html
    
    Parameters
    ----------
    dist : np.ndarray
        Distance matrix
    method : str
        Linkage method
  
    Returns
    -------
    np.ndarray
    """
    
    Z = linkage(dist, method)
    
    return Z

def get_clusters_from_linkage(Z, depth=None, no_clusters=None, t=1) -> np.ndarray:
    """
    Assign a cluster to each element from the linkage. Either depth or number of clusters needs to be specified.
    If depth is specified, criterion 'inconsistent' is used. If number of clusters is defined, 'maxclust' criterion is used.
    
    For description of the criterion check scipy documentation:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html
    
    Parameters
    ----------
    Z : np.ndarray
        Linkage
    depth : int
        The maximum depth to perform the inconsistency calculation.
    no_clusters : int
        The maximum number of clusters to be found.
    t : float
        Threshold to apply when forming flat clusters.
  
    Returns
    -------
    np.ndarray
    """
    
    if ((depth is not None and no_clusters is not None) or  (depth is None and no_clusters is None)):
        raise Exception("Either depth or no_clusters must be set.")
        
    if depth:
        return fcluster(Z, criterion="inconsistent", depth=depth, t=t)
    
    if no_clusters:
        return fcluster(Z, criterion="maxclust", t=no_clusters)
    
def get_embeddings_for(df: pd.DataFrame(), groupby: str, topic=None, characters=None, embedding_column="embeddings") -> pd.DataFrame():
    """
    Group data according to some column (e.g. Name) and compute the mean embedding over each group. 
    
    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe with text data and their embeddings.
    groupby : str
        The column in df by which the embeddings should be grouped together.
    topic : int
        Number of the topic to select.
    characters : list
        List of characters to select.
    embedding_column : str
        Name of the column in df containing the embeddings.
  
    Returns
    -------
    pd.DataFrame()
    """
    
    if characters is not None:
        df = df[df["Name"].isin(characters)]
     
    if topic is not None:
        df = df[df["Topics"] == topic]
    
    return df.groupby(groupby)[embedding_column].apply(lambda x: x.mean()).reset_index()

    
#----------- Argument Analysis -----------#

def plot_arguments(embeddings: pd.Series(), topic=None, clusters=None, annotations=None, just_plot=False):
    """
    Plot (character) embeddings using PCA projection into 2D space, optionally annotating each point and 
    visualising assignment of each point to a cluster.  
    
    Parameters
    ----------
    embeddings : pd.Series()
        Column (list) of embeddings
    topic : int
        Number of the selected topic for character embedding computing. Used only in the plot title.
    clusters : pd.Series()
        Cluster numbers corresponding to each embedding
    annotations : pd.Series()
        Annotations to be plotted for each embedding
    just_plot : boolean
        When you want to just plot the arguments/embeddings without any analysis or filtering.
  
    Returns
    -------
    None
    """
    
    sentences = list(embeddings)
    
    pca = PCA(n_components = 2)
    pc = pca.fit_transform(sentences)

    if clusters is None:
        plt.scatter(pc[:,0], pc[:,1])
    else:
        plt.scatter(pc[:,0], pc[:,1],  c=list(clusters), label=list(clusters))
        

    if annotations is not None:
        annotations = list(annotations)
        
        for i, s in enumerate(sentences):
            plt.annotate(annotations[i], (pc[i,0], pc[i,1]))
    if topic:
        plt.title(f"Visualisation of characters for Topic {topic}")
        
    elif just_plot:
        plt.title(f"Visualisation of embeddings")
        
    else:
        plt.title(f"Visualisation of characters overall and 2 clusters")
        
    plt.show()
    
    
def get_most_similar_characters(dist: np.ndarray, characters: list):
    """
    Gets 2 most similar characters and their similarity score from a distance matrix. 
    
    Parameters
    ----------
    dist : np.ndarray
        Distance matrix
    characters : list
        Character names corresponding to each row in the distance matrix
    Returns
    -------
    tuple
    """
    
    np.fill_diagonal(dist, np.inf)
    idx = np.unravel_index(dist.argmin(), dist.shape)
   
    return (characters[idx[0]], characters[idx[1]], dist.min())

def get_least_similar_characters(dist, characters):
    """
    Gets 2 least similar characters and their similarity score from a distance matrix. 
    
    Parameters
    ----------
    dist : np.ndarray
        Distance matrix
    characters : list
        Character names corresponding to each row in the distance matrix
    Returns
    -------
    tuple
    """
    
    np.fill_diagonal(dist, -1)
    idx = np.unravel_index(dist.argmax(), dist.shape)
    
    return (characters[idx[0]], characters[idx[1]], dist.max())

def topic_argument_analysis(df, topic, characters, embedding_column="d2v"):
    """
    If characters are provided, plots arguments and their cluster membership  for the selected characters, their distance matrix 
    and the 2 most and least similar characters. Always also plots the same statistics, but for 20 most common characters 
    in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame()
        DataFrame with the data and embeddings.
    topic : int
        Number of the selected topic for character embedding computing. Used only in the plot title.
    characters : list
        List of the characters to be selected for character embedding computing.
    embedding_column : str
        Name of the column in df containing the line embeddings.
    
    Returns
    -------
    None
    """
    
    # selected characters topic analysis
    if characters:
        embeddings = get_embeddings_for(df, "Name", topic, characters, embedding_column)
        print(embeddings)

        dist = cosine_dist(embeddings["d2v"])
        linkage = get_linkage(dist, "ward")
        embeddings['cluster'] = get_clusters_from_linkage(linkage, no_clusters=2)

        plot_arguments(embeddings["d2v"], topic=topic, annotations=embeddings["Name"], clusters=embeddings['cluster'])
        print(embeddings.groupby("cluster")['Name'].apply(lambda x: ','.join(x)).reset_index())


        cmap = sns.cm.rocket_r
        ax = sns.heatmap(1-dist, linewidth=0.5, cmap=cmap)        
        ax.set(title=f"Distance between selected characters for the topic {topic}")
        ax.set_xticklabels(embeddings["Name"], rotation=90)
        ax.set_yticklabels(embeddings["Name"], rotation=30)
        plt.title(f"Distance between selected characters for the topic {topic}")
        plt.show()

        print(f"{get_most_similar_characters(dist, embeddings['Name'])} are most similar characters in the topic {topic}.")
        print(f"{get_least_similar_characters(dist, embeddings['Name'])} are least similar characters in the topic {topic}.")

    # overall topic analysis
    top20_characters = df.groupby("Name")["Name"].count().sort_values(ascending=False).index[:20]
    
    
    embeddings = get_embeddings_for(df, "Name", topic, top20_characters, embedding_column)
    dist = cosine_dist(embeddings["d2v"])
    linkage = get_linkage(dist, "ward")
    embeddings['cluster'] = get_clusters_from_linkage(linkage, no_clusters=2)
    
    plot_arguments(embeddings["d2v"], topic=topic, annotations=embeddings["Name"], clusters=embeddings['cluster'])
    print(embeddings.groupby("cluster")['Name'].apply(lambda x: ','.join(x)).reset_index())

    
    cmap = sns.cm.rocket_r
    ax = sns.heatmap(1-dist, linewidth=0.5, cmap=cmap)
    ax.set_xticklabels(embeddings["Name"], rotation=90)
    ax.set_yticklabels(embeddings["Name"])
    plt.title(f"Similarity between top20_characters for the topic {topic}")
    plt.show()
    
    print(f"{get_most_similar_characters(dist, embeddings['Name'])} are most similar top20_characters in the topic {topic}.")
    print(f"{get_least_similar_characters(dist, embeddings['Name'])} are least similar top20_characters in the topic {topic}.")

def overall_argument_analysis(df, embedding_column="d2v"):
    """
    Plots arguments and their cluster membership for top 20 most common characters for the overall series (not just one topic).
    
    Parameters
    ----------
    df : pd.DataFrame()
        DataFrame with the data and embeddings.
    embedding_column : str
        Name of the column in df containing the line embeddings.
    
    Returns
    -------
    None
    """
    
    top20_characters = df.groupby("Name")["Name"].count().sort_values(ascending=False).index[:20]
    
    embeddings = get_embeddings_for(df, "Name", characters=top20_characters, embedding_column=embedding_column)
    dist = cosine_dist(embeddings["d2v"])
    linkage = get_linkage(dist, "ward")
    embeddings['cluster'] = get_clusters_from_linkage(linkage, no_clusters=2)
    
    plot_arguments(embeddings["d2v"], annotations=embeddings["Name"], clusters=embeddings['cluster'])
    print(embeddings.groupby("cluster")['Name'].apply(lambda x: ','.join(x)).reset_index())

    cmap = sns.cm.rocket_r
    ax = sns.heatmap(1-dist, linewidth=0.5, cmap=cmap)
    ax.set_xticklabels(embeddings["Name"], rotation=90)
    ax.set_yticklabels(embeddings["Name"])
    plt.title(f"Similarity between top20_characters for the overall series.")
    plt.show()
    
    print(f"{get_most_similar_characters(dist, embeddings['Name'])} are most similar characters in overall.")
    print(f"{get_least_similar_characters(dist, embeddings['Name'])} are least similar characters overall.")
    