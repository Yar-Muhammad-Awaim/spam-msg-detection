import re
import string

import demoji
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob

from utils.contractions import common_appos
from utils.slangs import common_slangs

lemmatizer = WordNetLemmatizer()


def remove_html_tags(msg: str) -> str:
    html_tags_pattern = re.compile(r"<.*?>")
    return html_tags_pattern.sub("", msg)


def remove_urls(msg: str) -> str:
    url_pattern = re.compile(
        r"\\b(?:(?:https?|ftp):\\/\\/)?(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)"
    )
    return url_pattern.sub("", msg)


def remove_stop_words(tokens: list[str]) -> list[str]:
    stop_words = stopwords.words("english")
    return [token for token in tokens if token not in stop_words]


def remove_punctuation(tokens: list[str]) -> list[str]:
    return [token for token in tokens if token not in string.punctuation]


def handle_appos(tokens: list[str]) -> list[str]:
    return [common_appos.get(token, token) for token in tokens]


def handle_slangs(tokens: list[str]) -> list[str]:
    return [common_slangs.get(token, token) for token in tokens]


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos="n") for token in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos="v") for token in tokens]
    return lemmatized_tokens


def correct_spellings(tokens: list[str]) -> str:
    msg = ""
    for token in tokens:
        msg += token

    text_blob = TextBlob(msg)
    text_blob = text_blob.correct()

    return text_blob


def remove_empty_tokens(tokens: list[str]) -> list[str]:
    return [token for token in tokens if len(token.strip()) > 0]


def pre_process_msg(df: pd.DataFrame, column: str) -> pd.DataFrame:
    # Lowercase messages
    df[column] = df[column].str.lower()

    # Convert emojis into their meanings
    df[column] = df[column].apply(demoji.replace_with_desc)

    # Remove html tags
    df[column] = df[column].apply(remove_html_tags)

    # Remove URLs
    df[column] = df[column].apply(remove_urls)

    # Tokenize the words
    df[column] = df[column].apply(word_tokenize)

    # Remove the stop words
    df[column] = df[column].apply(remove_stop_words)

    # Remove punctuation
    df[column] = df[column].apply(remove_punctuation)

    # Handle the appos
    df[column] = df[column].apply(handle_appos)

    # Handle common slangs
    df[column] = df[column].apply(handle_slangs)

    # Lemmatize tokens
    df[column] = df[column].apply(lemmatize_tokens)

    # Clean out any empty strings
    df[column] = df[column].apply(remove_empty_tokens)

    # Merge the tokens into strings
    df[column] = df[column].apply(lambda tokens: " ".join(tokens))

    # Correct spellings !!! TAKES A LOT OF TIME SO COMMENTING THIS.
    # print("Correcting spellings, it may take a while.")
    # df[column] = df[column].apply(correct_spellings)

    return df


def main():
    df = pd.read_csv("data/spam.csv", encoding="latin-1")
    df = pre_process_msg(df, "v2")
    df.to_csv("data/pre_processed_data.csv")


if __name__ == "__main__":
    main()
