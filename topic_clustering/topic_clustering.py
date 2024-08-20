from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram, fclusterdata, fcluster
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def get_tfidf_matrix(documents: list):
    """
    Computer tf-idf matrix of documents.
    
    Parameters
    ----------
    documents : list
        List of documents/lines.
    
    Returns
    -------
    scipy.sparse._csr.csr_matrix
    """
    
    tfidf_vectorizer = TfidfVectorizer(stop_words=None, use_idf=True, 
                                       tokenizer=None, ngram_range=(1,1))
        
    matrix = tfidf_vectorizer.fit_transform([" ".join(d) for d in documents])
    terms = tfidf_vectorizer.get_feature_names()

    return matrix

def cosine_dist(matrix) -> np.ndarray:
    """
    Computes pair-wise cosine distance between rows of a matrix.
    
    Parameters
    ----------
    matrix : scipy.sparse._csr.csr_matrix
        Matrix of vectorized documents.
  
    Returns
    -------
    np.ndarray
    """
    
    dist = 1-cosine_similarity(matrix)
    
    return dist

def get_linkage(dist, method='complete'):
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

def get_clusters_from_linkage(Z, depth=None, no_clusters=None, t=1):
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
    
def plot_clusters(data, clusters: pd.Series()):
    """
    Plots the data from a vectorized document matrix using SVD. Visualises clusters in the plot.
    
    Parameters
    ----------
    data : scipy.sparse._csr.csr_matrix
        Matrix of vectorized documents.
    clusters : pd.Series
        List of cluster corresponding to each document
  
    Returns
    -------
    None    
    """
    
    clf = TruncatedSVD(2)
    pca = clf.fit_transform(data)
    
    plt.scatter(pca[:,0], pca[:,1],c=clusters.astype("category"))    
    plt.show()
    
#----------- Cluster Analysis -----------#

def count_lines(df: pd.DataFrame()) -> int:
    """
    Counts lines (rows) in the dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe
  
    Returns
    -------
    int    
    """
    
    return df.shape[0]

def most_common_value(column: pd.Series(), n: int):
    """
    Returns n most common values in a column.
    
    Parameters
    ----------
    column : pd.Series()
        The column (list) to find the most common values in.
    n : int
        The number of most common values to find.
  
    Returns
    -------
    str    
    """
    
    counts = column.value_counts()
    
    if n > len(counts):
        n = len(counts)
    return " ,".join([counts.index[i] for i in range(n)])

def tf_idf_stats(df: pd.DataFrame(), cluster_column: str, interest_column: str, n: int):
    """
    Compute tf-idf for selected data (interest column), taking each cluster and data belonging to as one document.
    
    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe with the data
    cluster_column : str
        Name of the column in df containing the cluster number
    interest_column : str
        Name of the column for which tf-idf should be computed.
    n : int
        The number of entries with the highest tf-idf score to return for each cluster (document).
  
    Returns
    -------
    str    
    """
    
    clusters = df[cluster_column].unique()
    clusters.sort()
        
    vocabulary = df[interest_column].unique()
    documents = []
    
    for c in clusters:
        filtered = df[df[cluster_column] == c]
        
        document = " ".join([name if type(name) is str else '' for name in filtered["Name"]])
        documents.append(document)
        
    tfidf_vectorizer = TfidfVectorizer(stop_words=None, use_idf=True, 
                                       tokenizer=None, ngram_range=(1,1), vocabulary=vocabulary)
    
    matrix = tfidf_vectorizer.fit_transform(documents)
    feature_array = tfidf_vectorizer.get_feature_names()
    tfidf_sorting = np.argsort(matrix.toarray()).flatten()[::-1]
    
    dense = matrix.todense()
    denselist = dense.tolist()   
    
    tfidf_df = pd.DataFrame(denselist, columns=feature_array, index=clusters)
    
    top_n = {}
    
    for c in clusters:
        row = tfidf_df.loc[c].sort_values(ascending=False)
        
        top_n[c] = " ,".join([row.index[i] for i in range(n)])
    
    return top_n

def compute_statistics(df: pd.DataFrame(), tfidf: dict, group: str, n):
    """
    Using the datafram with data and a tfidf matrix, compute the number of lines, most common characters, 
    most common series, most significant tfidf characters and most significant tfidf seasons for the specified cluster 
    in the data.
    
    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe with data
    tfidf : dict
        Dictionary containing tfidf matrix for characters and series
    group : str
        Name of the cluster to compute statistics for
    n : int
        The number of most common and significant values to find.
  
    Returns
    -------
    pd.Series()    
    """
    
    summary = {}
    
    summary["lines count"] = count_lines(df)
    summary["top character"] = most_common_value(df["Name"], n)
    summary["top series"] = most_common_value(df["Season"], n)
    summary["tfidf character"] = tfidf["characters"][group]
    summary["tfidf season"] = tfidf["seasons"][group]

    
    return pd.Series(summary)

def analyze(df: pd.DataFrame(), group: str, n=3) -> pd.DataFrame():
    """
    Computes statistics for each group (cluster) and puts the results into one dataframe.
    
    Parameters
    ----------
    df : pd.DataFrame()
        Dataframe with data
    group : str
        Name of the column containing the grouping criteria (cluster number).
    n : int
        The number of most common and significant values to find.
  
    Returns
    -------
    pd.DataFrame()    
    """
    
    g = df.groupby(group)
    tfidf = {}
    
    tfidf["characters"] = tf_idf_stats(df, group, "Name", n)
    tfidf["seasons"] = tf_idf_stats(df, group, "Season", n)
    
    print(f"Number of clusters: {len(df[group].unique())}")
    
    return g.apply(lambda x, tfidf=tfidf, n=n: compute_statistics(x, tfidf, x.name, n)).sort_values("lines count", ascending=False)

