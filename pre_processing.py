"""
### Preprocess lines for topic modelling in ONECELL
import pre_processing as pp
from utils import get_df

df = get_df()

tokenizer = pp.TokTok_Tokenizer()

stopwords = pp.stopwords()

translator = pp.translator()

#nltk.download('wordnet')
lemmatizer = pp.lemmatizer()

stemmer = pp.stemmer()

df = df.replace({"Name": pp.corrected_names_dict()})


df["Tokenized"] = df["Sentence"].map(lambda x: tokenizer.tokenize(x))
df["Cleaned"] = df["Tokenized"].map(lambda x: pp.clean_tokens(x,translator,stopwords))
#df["Stemmed"] = df["Cleaned"].map(lambda x: pp.stem(x, stemmer))
df["Lemmatized"] = df["Cleaned"].map(lambda x: pp.lemmatize(x, lemmatizer))
df["PoS Tagged"] = df["Lemmatized"].map(lambda x: pp.pos_tag(x))
df["Nouns only"] = df["PoS Tagged"].map(lambda x: pp.get_pos_tagged_words(x, ['NN']))
df["Nouns and Verbs"] = df["PoS Tagged"].map(lambda x: pp.get_pos_tagged_words(x, ['NN', 'VB']))
df["Nouns, Adjectives and Verbs"] = df["PoS Tagged"].map(lambda x: pp.get_pos_tagged_words(x, ['NN', 'JJ', 'VB']))
df["Nouns, Adjectives, Adverbs and Verbs"] = df["PoS Tagged"].map(lambda x: pp.get_pos_tagged_words(x, ['NN', 'JJ', 'RB', 'VB']))
"""


"""
Contains methods used for pre-processing the data.
To be called from the pipeline.ipynb file.
"""

import nltk
import string
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import *


### Name Unification ###
def corrected_names_dict():
    """
    Unifies the names in the dataset that are duplicated   

    Returns
    -------
    Dict 
    """

    return {'alliser': 'alliser thorn', 'alliser thorne': 'alliser thorn', 'benjen': 'benjen stark',
            'deanerys targarian': 'daenerys targaryen', 'ed': 'edd', 'janos': 'janos slynt', 'jeor': 'jeor mormont',
            'kevan': 'kevan lannister', 'khal': 'khal drogo', 'king joffrey': 'joffrey lannister', 'kraznys': 'kraznys mo nakloz',
            'lancel': 'lancel lannister','loras': 'loras tyrell', 'lyann': 'lyanna', 'lysa': 'lysa arryn', 'meryn': 'meryn trant',
            'petyr': 'petyr baelish', 'pyatt pree': 'pyat pree', 'pyp': 'pypar', 'renly': 'renly baratheon', 'rikon': 'rickon',
            'robett': 'robert baratheon', 'robin': 'robin arryn', 'rodrik': 'rodrik cassel', 'roose': 'roose bolton',
            'roz': 'ros', 'sam': 'sam tarly', 'sammy': 'sam tarly', 'sandor': 'sandor clegane', 'walder': 'walder frey',
            'yohn': 'yohn royce', 'young ned': 'eddard stark'}



### Deleting Dothraki
def delete_dothraki(line, tokenizer, checks=False):
    """
    Deletes the lines that include dothraki language from the dataset.
    
    Parameters
    ----------
    line : String
        A line from the dataset
    tokenizer : tokenizer Object
        Chosen tokenizer
    checks : Boolean
        For debugging purposes while adding new langugages
  
    Returns
    -------
    String    
    """


    dothraki_dict = dict()

    with open("data/dothraki.txt") as f:
        for i, dothraki_line in enumerate(f.read().split("\n")):
            if (i > 62)& (i < 2572): # where the word definitions are in the txt file
                if ":" in dothraki_line:
                    word = dothraki_line.partition(":")[0] # https://stackoverflow.com/questions/8162021/analyzing-string-input-until-it-reaches-a-certain-letter-on-python
                    word = word.partition(" ")[0]
                    dothraki_dict.update({i: word})

    if checks:
        #print(dothraki_dict.values())
        #print(dothraki_dict[2571]) # last word 
        print('zorat:', 'zorat' in dothraki_dict.values()) 
        print('jon: ', 'jon' in dothraki_dict.values()) 
        print(line)

    correlating_words = ['jon', 'khaleesi', 'save', 'fire', 'ale', 
                         'me', 'at', 'has', 'she', 'rich'] #correlating words after tokenizing

    dothraki_dict = {k: v for k, v in dothraki_dict.items() if v not in correlating_words}

    if checks:
        print('jon: ', 'jon' in dothraki_dict.values()) 

    tokens = tokenizer.tokenize(line)
    translator = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(translator) for token in tokens]
    if checks:
        print(tokens)

    for word in tokens:
        if word in dothraki_dict.values():
            if checks:
                print(word)  
            line = ""
    
    if checks:
        print(line)    
    return line

    

    

    



### Tokenizers ###

def TokTok_Tokenizer(example_sentence=False):
    """
    Loads TokTok Tokenizer
    
    Parameters
    ----------
    example_sentence : Boolean
        Example to make tokenizer choosing easier
  
    Returns
    -------
    tokenizer Object    
    """
    toktok = ToktokTokenizer()
    if example_sentence:
        print("Example of tokenization:")
        print(toktok.tokenize(example_sentence))
    return toktok


def Nltk_Tokenizer(example_sentence=False):
    """
    NLtk Tokenizer
    
    Parameters
    ----------
    example_sentence : Boolean
        Example to make tokenizer choosing easier
  
    """
    print("No initialization needed for nltk tokenizer. Usage: 'nltk.word_tokenize('String')' ")
    if example_sentence:
        print("Example of tokenization:")
        print(nltk.word_tokenize(example_sentence))




### Cleaners ###
def stopwords(show=False):
    """
    Defines and extends the english stopwords from nltk

    More alternatives: 
    https://gist.github.com/sebleier/554280
    https://countwordsfree.com/stopwords 
    
    Parameters
    ----------
    show : Boolean
        Shows the modified stopword list
  
    Returns
    -------
    List    
    """

    stopwords = nltk.corpus.stopwords.words('english')
    # extracted meaningless 2-3 letters words from dataset 
    # use extract_tokens_w_len_123() to see the remaning 2-3 letter words
    stopwords.extend(['u', '…',  'ye', 'lt' ,'eh', 'em', 'oh', 'mm', 'hm', 'ya', 'ed', 'ah', 'oi', 'wa', 'mo', 
                  'lf', 'ls', 'uh', 'le', 'ta', 'er', 'ha', 'ho', 'ow', 'ch', 'ii', 'ox', 'jo', 'lo', 
                  'th', 'un', 'al', 'zo', 'um', 'ti', 'ay', 'na', 
                  'ooh', 'mmh', 'mmm', 'huh', 'hmm', 'shh', 'wyl', 'tis', '‘em', 'wel', 'ahh', 
                  'mra', 'yah', 'nan', 'hhe', 'toi', 'aha', 'iti', 'nic', 'sno', 'rye', 'nah', 'tho',
                  'heh', 'inv', 'nig', 'aah', 'bah', 'hin', 'bur', 'iit', 'kif', 'wun', 'wan', 'ary',
                  'yeah', 'uhhuh'])
    if show:
        print("Extracted meaningless 2-3 letters words from dataset: ")
        print(stopwords)
    return stopwords


def extract_tokens_w_len_123(lines):
    """
    Function to manually check the remaining possible stopwords
    or meaningless syllables emerging after tokenizing

    Shows three lists of 1,2,3 letter syllables
    
    Parameters
    ----------
    line : String
        A line from the dataset  
    """

    one_letter = []
    two_letters = []
    three_letters = []

    for line in lines:
        for word in line:
            if (len(word) < 2) & (word not in one_letter):
                one_letter.append(word)
            if (len(word) < 3) & (word not in two_letters):
                two_letters.append(word) 
            if (len(word) > 2) & (len(word) < 4) & (word not in three_letters):
                three_letters.append(word) 

    print(sorted(one_letter))
    print(sorted(two_letters))
    print(sorted(three_letters))


def translator():
    """
    Defines a translator that sets ASCII IDs of puntuations to None
  
    Returns
    -------
    translator Object    
    """
    translator = str.maketrans('', '', string.punctuation)
    return translator


def clean_tokens(line,translator,stopwords):
    """
    The function performs the following in order:
        remove punctuation
        apply lowercase the tokens
        remove stopwords
        remove empty items 
        remove 1 letter tokens ('i' and 'a' are already in stopwords)
        take the set of tokens
    
    Parameters
    ----------
    line : String
        A line from the dataset 
    translator : translator Object
        Used to remove punctuations
    stopwords : List
        Predefined stopwords to erase from the lines

    

    Returns
    -------
    List 
    """

    tokens = [token.lower() for token in line] # tokenizing and converting to lowercase
    tokens = [token.translate(translator) for token in tokens] # removing punctuation characters
    tokens = [token for token in tokens if token not in stopwords]  # removing stopwords
    tokens = [token for token in tokens if len(token) >1] #removing empty tokens
    tokens = list(set(tokens))
   
    return tokens



### Stemmers ###
def lemmatizer():
    """
    Initializes the WordNetLemmatizer
  
    Returns
    -------
    translator Object    
    """
    lemmatizer = WordNetLemmatizer()
    return lemmatizer


def stemmer():
    """
    Initializes the PorterStemmer
  
    Returns
    -------
    stemmer Object    
    """
    stemmer = PorterStemmer()
    return stemmer


def lemmatize(line, lemmatizer):
    """
    Appilies lemmatization 

    Parameters
    ----------
    line : String
        A line from the dataset 
    lemmatizer : lemmatizer Object
        Predefined lemmatizer

    Returns
    -------
    translator Object    
    """
    tokens = [word for word in line]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens


def stem(line, stemmer):
    """
    Appilies stemming 
    
    Parameters
    ----------
    line : String
        A line from the dataset 
    stemmer : stemmer Object
        Predefined stemmer

    Returns
    -------
    translator Object    
    """
    tokens = [word for word in line]
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens



### POS Tagging ###
def pos_tag(line):
    """
    Appilies Part of Speech tagging to the given tokens 
    
    Parameters
    ----------
    line : String
        A line from the dataset 

    Returns
    -------
    List    
    """
    tokens = [word for word in line]
    tokens = nltk.pos_tag(tokens)
    return tokens


def get_pos_tagged_words(line, pos_tag_list="default"):
    """
    Appilies Part of Speech tagging to the given tokens 
    
    Parameters
    ----------
    line : String
        A line from the dataset 
    pos_tag_list : List
        List of PoS Tag types to to retrieve by choice
        (see below for default version)

    Returns
    -------
    List    
    """
    if pos_tag_list=="default":
        print("No tags given using default tags: ['NN', 'VB']")
        pos_tag_list = ['NN', 'VB']
    tokens = [word for word in line]
    tokens = [token[0] for token in tokens if token[1] in pos_tag_list]
    return tokens
