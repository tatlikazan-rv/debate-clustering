import csv
import re
import string
import os
import pandas as pd
from typing import List

import nltk

nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
FILE_NAME = "data/Game_of_Thrones_Script.csv"
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(CURRENT_DIR, FILE_NAME)


def get_df() -> pd.DataFrame:
    """
    returns Dataframe from FILE_PATH
    :return:
    """
    return pd.read_csv(FILE_PATH)


def data_lines() -> List[str]:
    """
    uses FILE_NAME to return a list of sentences
    :return:
    """
    with open(FILE_NAME, newline='', encoding="utf8") as csvfile:
        reader = csv.reader(csvfile)
        sentences = []

        for row in reader:
            text = row[5]
            text = text.lower()
            text = re.sub('[%s]' % re.escape(string.punctuation), ' ', text)
            text = re.sub('�', ' ', text)
            sentences.append(text)
    return sentences