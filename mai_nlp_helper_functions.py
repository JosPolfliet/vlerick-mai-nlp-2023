

import datetime
import string

import numpy as np
import pandas as pd
import spacy
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.vectors import Vectors
from tabulate import tabulate
from tqdm import tqdm

tqdm.pandas()

punctuations = string.punctuation

nlp = spacy.load('en_core_web_md')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = nlp(sentence["text"].lower())

    # Remove OOV words
    mytokens = [ word for word in mytokens if not word.is_oov ]
    
    # Lemmatise + lower case
    mytokens = [ word.lemma_.strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Remove stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    return mytokens



def log_experiment_results(experiment_name, stats, filename="experiment_log.md"):
    """
    Appends experiment results and statistics to a markdown log file.
    
    Parameters:
    - experiment_name: str, the name of the experiment
    - stats: dict, a dictionary containing the statistics to log
    - filename: str, the path to the log file
    """
    stats["timestamp"] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    stats["Experiment Name"] = experiment_name
    try:
        
        df = pd.read_table(filename, sep="|", skipinitialspace=True).drop(0)
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame(columns=list(stats.keys()))
    
    df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
    df = df[["precision", "recall", "f1-score", "support", "timestamp", "Experiment Name"]]
    markdown_table = tabulate(df, headers='keys', tablefmt='pipe', showindex=False, floatfmt=(".3g"), intfmt=",")
    with open(filename, 'w') as f:
        f.write(markdown_table)

def evaluate_model(y_test, predictions, classes):
    stats = classification_report(y_test, predictions, output_dict=True)
    print(classification_report(y_test, predictions))
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8, 5))
    
    cmp = ConfusionMatrixDisplay(
        confusion_matrix(y_test, predictions),
        display_labels=classes,
    )
    
    cmp.plot(ax=ax)
    plt.show()
    return stats