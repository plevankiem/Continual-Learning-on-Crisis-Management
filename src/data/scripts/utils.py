import torch
import torch.nn as nn
import plotly.graph_objects as go
from IPython.display import display
import numpy as np
import pandas as pd
import os
import glob
import gc
import json
from time import time
from transformers import BertTokenizer, BertTokenizerFast, BertModel, FlaubertModel, FlaubertTokenizer
from tqdm.auto import tqdm
from collections import deque, namedtuple
import random
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from itertools import islice
import re
import sacremoses

class TextPreprocessing:
    def __init__(self):
        self.tokenizer = tokenizer
    
        def remove_cities(text):
            """
            Remove cities from text
            :param text: text to remove cities from
            :returns: texte without cities
            """
            crisis_names = ['irma','bruno','aude','harvey','eleanor','corse-fione','beryl−guadeloupe','corse','egon','susanna','ulrika','reunion−berguitta','marseille','effondrementmarseille','guadeloupe','corse','immeuble','martinique','saint martin','berguitta']
            crisis_scrap = ['marseille','bruno','crue', 'crues', 'aude', 'carcassonne', 'trèbes', 'trebes','corse', 'corsica', 'hautecorse', 'haute-corse','crue','béryl', 'beryl', 'guadeloupe', 'ondetropicale','réunion', 'reunion', 'lareunion', 'fakir', 'laréunion','réunion', 'reunion', 'lareunion',' berguitta',' laréunion','corse', 'fionn', 'corsica', 'ana','irma','ouraganIRMA', 'saintmartin', 'stmartin', 'saintbarthelemy', 'saintbarth', 'stbarth','harvey', 'martinique', 'guadeloupe','egon','ulrika', 'vendée','bretagne','susanna']
            crisis_scrap=crisis_scrap+crisis_names
            big_regex = re.compile('|'.join(map(re.escape, crisis_scrap)))
            text = big_regex.sub(" ", text)
            return text
        
        def remove_url(text):
            """
            Remove URL from text
            :param texte: text to remove URL from
            :returns: texte without URL
            """
            text = re.sub(r'http(\S)+', '', text)
            text = re.sub(r'http ...', '', text)
            return text

        def remove_rt(text):
            """
            Remove RT mention from text
            :param text: text to remove rt mention from
            :returns: texte without rt mention
            """
            return re.sub(r'(RT|rt)[ ]*@[ ]*[\S]+', '', text)

        def remove_at(text):
            """
            Remove at mention from text
            :param text: text to remove ata mention from
            :returns: texte without at mention
            """
            return re.sub(r'@\S+', '', text)

        def remove_extraspace(text):
            """
            Remove extra space from text
            :param text: text to remove rt mention from
            :returns: texte without rt mention
            """
            return re.sub(r' +', ' ', text)

        def replace_and(text):
            """
            Replace &amp which represents and in html with and
            :param text: text to replace from
            :returns: texte and replaced
            """
            return re.sub(r'&amp;?', 'and', text)

        def replace_lt_lg(text):
            """
            replace lt and lg that stands for lower and upper in html with proper signs
            :param text: text to replace lt and lg in
            :returns: texte without &lt and &lg
            """
            text = re.sub(r'&lt;', '<', text)
            text = re.sub(r'&gt;', '>', text)
            return text

        def lower(text):
            """
            lowercase text
            :param text: text to lowercase
            :returns: texte lowercased
            """
            return text.lower()

        def lower_then_k(text, k=3):
            """
            asserts text length
            :param text: text to mesure length
            :returns: null if text length lower then k, text instead
            """
            return len(text) > k

        def remove_numero(text):
            """
            replace numbers from and replace them with numero
            :param text: text
            :returns: texte without numbers
            """
            return re.sub(r'\d+', 'numero', text)

        def remove_punctuations(text):
            """
            remove ponctuations
            :param text: text
            :returns: text without punctuations
            """

            return re.sub('["$#%()*+,-@./:;?![\]^_`{|}~\n\t’\']', ' ', text)

        def remove_pic_tweet(text):
            """
            remove pic from tweet
            :param text :
            :return : text without pic
            """
            return re.sub(r'pic.twitter.com(.*?)\s(.*)', '', text)
        
        self.pretraitement = [
            remove_url,
            remove_rt,
            remove_at,
            replace_and,
            replace_lt_lg,
            remove_numero,
            remove_punctuations,
            remove_extraspace,
            remove_pic_tweet,
            remove_cities
        ]

    def clean_tweet(self, text):
        for func in self.pretraitement:
            text = func(text)
        return text

    def preprocess_text(self, text_series):
        text_series = text_series.apply(self.clean_tweet)
        return text_series

class BertInput():
    def __init__(self, Tokenizer, max_length):
        self.tokenizer = Tokenizer
        self.max_length = max_length

    def encode_sents(self, sentences):
        """ Tokenize list of sentences according do the tokenizer provided
        @params sents (list[sentences]):list of sentences , where each sentence is represented as a sting
        @params tokenizer (PretrainedTokenizer): Tokenizer that tokenizes text
        @returns
        """
        # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []

        # For every sentence...
        for sent in sentences:
            # `encode` will:
            #   (1) Tokenize the sentence.
            #   (2) Prepend the `[CLS]` token to the start.
            #   (3) Append the `[SEP]` token to the end.
            #   (4) Map tokens to their IDs.
            encoded_sent = self.tokenizer.encode(sent, truncation=True, max_length = self.max_length)

            # Add the encoded sentence to the list.
            input_ids.append(encoded_sent)
        return input_ids

    def pad_sents(self, sents):
        """ Pad list of sentences according to the longest sentence in the batch.
        @param sents (list[list[int]]): list of sentences, where each sentence
                                        is represented as a list of words
        @param pad_token (int): padding token
        @returns sents_padded (list[list[int]]): list of sentences where sentences shorter
            than the max length sentence are padded out with the pad_token, such that
            each sentences in the batch now has equal length.
            Output shape: (batch_size, max_sentence_length)
        """
        sents_padded = []
        batch_size = len(sents)

        for s in sents:
            padded = [self.tokenizer.pad_token_id] * self.max_length
            padded[:len(s)] = s
            sents_padded.append(padded)

        return sents_padded

    def mask_sents(self, input_ids):
        attention_masks = []
        """ mask sentences
        :param text: input_ids
        :returns: masked sentences
        """

        # For each sentence...
        for sent in input_ids:

            # Create the attention mask.
            #   - If a token ID is 0, then it's padding, set the mask to 0.
            #   - If a token ID is > 0, then it's a real token, set the mask to 1.
            att_mask = [int(token_id != self.tokenizer.pad_token_id)
                        for token_id in sent]

            # Store the attention mask for this sentence.
            attention_masks.append(att_mask)
        return attention_masks

    def fit_transform(self, sents, max_length = 146):
        """ get input_ids and mask_ids
        :param sents: list of sentences
        :returns: input ids and masks
        """
        input_ids = self.encode_sents(sents)
        input_ids = self.pad_sents(input_ids)
        mask = self.mask_sents(input_ids)
        return [ torch.tensor(input_ids), torch.tensor(mask) ]