# SPDX-FileCopyrightText: Copyright (c) James Bradbury and Soumith Chintala 2016
# SPDX-FileCopyrightText: Copyright (C) 2016-2024 ExplosionAI GmbH, 2016 spaCy GmbH, 2015 Matthew Honnibal
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause AND MIT

import re

# Adapted from https://github.com/explosion/spaCy/blob/32c4b638ae0e042264fadc3ceb3ab67147c7b6f5/spacy/lang/en/stop_words.py
ENGLISH_STOP_WORDS = {
    'those', 'on', 'own', '’ve', 'yourselves', 'around', 'between', 'four',
    'been', 'alone', 'off', 'am', 'then', 'other', 'can', 'regarding',
    'hereafter', 'front', 'too', 'used', 'wherein', '‘ll', 'doing',
    'everything', 'up', 'onto', 'never', 'either', 'how', 'before', 'anyway',
    'since', 'through', 'amount', 'now', 'he', 'was', 'have', 'into', 'because',
    'not', 'therefore', 'they', 'n’t', 'even', 'whom', 'it', 'see', 'somewhere',
    'thereupon', 'nothing', 'whereas', 'much', 'whenever', 'seem', 'until',
    'whereby', 'at', 'also', 'some', 'last', 'than', 'get', 'already', 'our',
    'once', 'will', 'noone', "'m", 'that', 'what', 'thus', 'no', 'myself',
    'out', 'next', 'whatever', 'although', 'though', 'which', 'would',
    'therein', 'nor', 'somehow', 'whereupon', 'besides', 'whoever', 'ourselves',
    'few', 'did', 'without', 'third', 'anything', 'twelve', 'against', 'while',
    'twenty', 'if', 'however', 'herself', 'when', 'may', 'ours', 'six', 'done',
    'seems', 'else', 'call', 'perhaps', 'had', 'nevertheless', 'where',
    'otherwise', 'still', 'within', 'its', 'for', 'together', 'elsewhere',
    'throughout', 'of', 'others', 'show', '’s', 'anywhere', 'anyhow', 'as',
    'are', 'the', 'hence', 'something', 'hereby', 'nowhere', 'latterly', 'say',
    'does', 'neither', 'his', 'go', 'forty', 'put', 'their', 'by', 'namely',
    'could', 'five', 'unless', 'itself', 'is', 'nine', 'whereafter', 'down',
    'bottom', 'thereby', 'such', 'both', 'she', 'become', 'whole', 'who',
    'yourself', 'every', 'thru', 'except', 'very', 'several', 'among', 'being',
    'be', 'mine', 'further', 'n‘t', 'here', 'during', 'why', 'with', 'just',
    "'s", 'becomes', '’ll', 'about', 'a', 'using', 'seeming', "'d", "'ll",
    "'re", 'due', 'wherever', 'beforehand', 'fifty', 'becoming', 'might',
    'amongst', 'my', 'empty', 'thence', 'thereafter', 'almost', 'least',
    'someone', 'often', 'from', 'keep', 'him', 'or', '‘m', 'top', 'her',
    'nobody', 'sometime', 'across', '‘s', '’re', 'hundred', 'only', 'via',
    'name', 'eight', 'three', 'back', 'to', 'all', 'became', 'move', 'me', 'we',
    'formerly', 'so', 'i', 'whence', 'under', 'always', 'himself', 'in',
    'herein', 'more', 'after', 'themselves', 'you', 'above', 'sixty', 'them',
    'your', 'made', 'indeed', 'most', 'everywhere', 'fifteen', 'but', 'must',
    'along', 'beside', 'hers', 'side', 'former', 'anyone', 'full', 'has',
    'yours', 'whose', 'behind', 'please', 'ten', 'seemed', 'sometimes',
    'should', 'over', 'take', 'each', 'same', 'rather', 'really', 'latter',
    'and', 'ca', 'hereupon', 'part', 'per', 'eleven', 'ever', '‘re', 'enough',
    "n't", 'again', '‘d', 'us', 'yet', 'moreover', 'mostly', 'one', 'meanwhile',
    'whither', 'there', 'toward', '’m', "'ve", '’d', 'give', 'do', 'an',
    'quite', 'these', 'everyone', 'towards', 'this', 'cannot', 'afterwards',
    'beyond', 'make', 'were', 'whether', 'well', 'another', 'below', 'first',
    'upon', 'any', 'none', 'many', 'serious', 'various', 're', 'two', 'less',
    '‘ve'
}
_patterns = [
    r"\'", r"\"", r"\.", r"<br \/>", r",", r"\(", r"\)", r"\!", r"\?", r"\;",
    r"\:", r"\s+"
]
_replacements = [" ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " ", " "]
_patterns_dict = list(
    (re.compile(p), r) for p, r in zip(_patterns, _replacements))


class Tokenizer:
    r"""Singleton wrapper for tokenizers to avoiding re-loading it multiple
    times."""
    def __new__(cls):
        if not hasattr(cls, 'tokenizer'):
            cls.tokenizer = _basic_english_normalize

        return cls.tokenizer


# Code adapted from:
# https://github.com/pytorch/text/blob/9bed85d7a7ae13cf8c28598a88d8e461fe1afcb4/torchtext/data/utils.py#L17-L49
def _basic_english_normalize(line):
    r"""
    Basic normalization for a line of text.
    Normalization includes
    - lowercasing
    - complete some basic text normalization for English words as follows:
        add spaces before and after '\''
        remove '\"',
        add spaces before and after '.'
        replace '<br \/>'with single space
        add spaces before and after ','
        add spaces before and after '('
        add spaces before and after ')'
        add spaces before and after '!'
        add spaces before and after '?'
        replace ';' with single space
        replace ':' with single space
        replace multiple spaces with single space

    Returns a list of tokens after splitting on whitespace.
    """
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line.split()
