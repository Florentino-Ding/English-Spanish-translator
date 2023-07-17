import numpy as np
import pandas as pd


def data_preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    para: data: pd.DataFrame, the data should contain four column, 0, 3 is the
        id of the sentence, 1 is the English sentence, 2 is the Spanish sentence
    return: pd.DataFrame, ie. the data after preprocessing, the dtype of the
        data is string

    This function is used to preprocess the data, it will do the following things:
        1, replace the upper case letter with lower case letter
        2, add space between the punctuation and the word
        3, replace the non-breaking space with space
    """

    def add_space(sentence: str) -> str:
        obj = str()
        for i, char in enumerate(sentence):
            if char in [
                ",",
                ".",
                "?",
                "!",
                ":",
                ";",
                "'",
                '"',
                "(",
                ")",
                "¡",
                "¿",
            ]:
                if i == 0:
                    obj += char + " "
                elif i == len(sentence) - 1:
                    obj += " " + char
                else:
                    obj += " " + char + " "
            else:
                obj += char
        return obj

    data.drop(columns=[0, 2], inplace=True)
    data = data.dropna()
    for col in data.columns:
        # 用小写字母替换大写字母
        data.loc[:, col] = data[col].str.lower()
        # 去除不间断空格
        data.loc[:, col] = data[col].apply(lambda x: x.replace("\u202f", " "))
        data.loc[:, col] = data[col].apply(lambda x: x.replace("\xa0", " "))
        # 在标点符号与单词之间加空格
        data.loc[:, col] = data[col].apply(add_space)
    data = data.astype("string")
    return data


def tokenlize(data: pd.DataFrame) -> tuple:
    """
    para: data: pd.DataFrame, the data should contain two column, English and
        Spanish
    return: tuple, ie. (eng, spa), eng and spa are list, the element of the
        list is the tokenlized sentence

    each sentence in the input data should be a string, and the sentence should be tokenlized by space
    """
    eng, spa = list(), list()
    for _, row in data.iterrows():
        eng.append(row["English"].split(" "))
        spa.append(row["Spanish"].split(" "))
    return eng, spa


def vocablize(
    data: list[list], min_freq=2, reserved_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]
) -> tuple:
    """
    para: data: list[list], the element of the list is the tokenlized sentence
        min_freq: int, the minimum frequency of the word, if the frequency of
        the word is less than min_freq, the word will be deleted
        reserved_tokens: list, the reserved tokens, the index of the reserved tokens will be 0, 1, 2, ...
    return: tuple, ie. (word2idx, idx2word), word2idx is a dict, the key is
        the word, the value is the index of the word, idx2word is a dict, the key is the index of the word, the value is the word
    """
    word2idx, idx2word = dict(), dict()
    word_freq = dict()
    counter = len(reserved_tokens)
    for i, token in enumerate(reserved_tokens):
        word2idx[token] = i
    for sentence in data:
        for word in sentence:
            if word not in word2idx:
                word2idx[word] = counter
                counter += 1
            word_freq[word] = word_freq.get(word, 0) + 1
    for word, freq in word_freq.items():
        if freq < min_freq:
            del word2idx[word]
    for word, idx in word2idx.items():
        idx2word[idx] = word
    return word2idx, idx2word


def string2idx(data: list[list], word2idx: dict) -> np.ndarray:
    """
    para: data: list[list], the element of the list is the tokenlized sentence
        word2idx: dict, the key is the word, the value is the index of the word
    return: np.ndarray, the dtype of the array is int, the shape of the array
        is (len(data), max_len)

    this function is used to convert the sentence to the index
    """
    sentence_len = [max([len(sentence) for sentence in data])]
    sentence_num = len(data)
    result = np.zeros((sentence_num, sentence_len), dtype=np.int64)
    for i, sentence in enumerate(data):
        for j, word in enumerate(sentence):
            result[i, j] = word2idx.get(word, word2idx["<unk>"])
    return result
