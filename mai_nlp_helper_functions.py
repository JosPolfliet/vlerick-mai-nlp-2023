import datetime
import string

import pandas as pd
import spacy
from matplotlib import pyplot as plt
from sklearn.metrics import (ConfusionMatrixDisplay, classification_report,
                             confusion_matrix)
from tabulate import tabulate
from tqdm import tqdm

tqdm.pandas()

punctuations = string.punctuation
nlp = None
stop_words = None

def spacy_tokenizer(sentence):
    """
    Tokenises a sentence using spaCy.
    Parameters:
    - sentence: str, the sentence to tokenise
    Returns:
    - mytokens: list, the list of tokens
    """
    # Creating our token object, which is used to create documents with linguistic annotations.
    global nlp
    global stop_words
    if not nlp:
        try:
            nlp = spacy.load("en_core_web_md")
            stop_words = spacy.lang.en.stop_words.STOP_WORDS
        except:
            spacy.cli.download("en_core_web_md")
            nlp = spacy.load("en_core_web_md")
            stop_words = spacy.lang.en.stop_words.STOP_WORDS

    tokens = nlp(sentence["text"].lower())

    # Remove OOV words
    tokens = [word for word in tokens if not word.is_oov]

    # Lemmatise + lower case
    tokens = [
        word.lemma_.strip() if word.lemma_ != "-PRON-" else word.lower_
        for word in tokens
    ]

    # Remove stop words
    tokens = [
        word for word in tokens if word not in stop_words and word not in punctuations
    ]

    return tokens


def log_experiment_results(experiment_name, stats, filename="experiment_log.md"):
    """
    Appends experiment results and statistics to a markdown log file.

    Parameters:
    - experiment_name: str, the name of the experiment
    - stats: dict, a dictionary containing the statistics to log
    - filename: str, the path to the log file
    """
    stats["timestamp"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    stats["Experiment Name"] = experiment_name
    try:

        df = pd.read_table(filename, sep="|", skipinitialspace=True).drop(0)
        df.columns = df.columns.str.strip()
        df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError):
        df = pd.DataFrame(columns=list(stats.keys()))

    df = pd.concat([df, pd.DataFrame([stats])], ignore_index=True)
    df = df[
        ["precision", "recall", "f1-score", "support", "timestamp", "Experiment Name"]
    ]
    markdown_table = tabulate(
        df,
        headers="keys",
        tablefmt="pipe",
        showindex=False,
        floatfmt=(".3g"),
        intfmt=",",
    )
    with open(filename, "w") as f:
        f.write(markdown_table)


def evaluate_model(y_test, predictions, classes):
    """
    Prints classification report and confusion matrix.

    Parameters:
    - y_test: list, the true labels
    - predictions: list, the predicted labels
    - classes: list, the list of classes

    Returns:
    - stats: dict, the classification report
    """
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
