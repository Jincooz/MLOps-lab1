import string

import re
USER_RE = re.compile(r"@\w+") # mentions
LINK_RE = re.compile(r"http\S+") # URLs
HTML_ENTITY_RE = re.compile(r"&#\w+;|&\w+;") # html tags
HASHTAG_RE = re.compile(r"#\w+") # hashtags
CLEAN_RE = re.compile(r"[^a-zA-Z<>|\s]") # punctuation except for <>
SPACE_RE = re.compile(r"\s+")
NUMBERS_RE = re.compile(r"\d+") # numbers


class TextPreprocessor:

    def __init__(self, config: dict = None):
        self.config = config or {
            "lowercase": True,
            "replace_mentions": True,
            "replace_urls": True,
            "remove_html": True,
            "remove_hashtags": True,
            "remove_punctuation": True,
            "remove_extra_whitespace": True,
            "remove_numbers" : False,
            "min_length": 2
        }

    def transform(self, texts):
        if isinstance(texts, str):
            return self._clean(texts)
        return [self._clean(t) for t in texts]

    def _clean(self, text: str) -> str:

        cfg = self.config


        if cfg["replace_mentions"]:
            text = USER_RE.sub("<USER>", text)

        if cfg["replace_urls"]:
            text = LINK_RE.sub("<LINK>", text)

        if cfg["remove_html"]:
            text = HTML_ENTITY_RE.sub("", text)

        if cfg["remove_hashtags"]:
            text = HASHTAG_RE.sub("", text)

        if cfg["remove_numbers"]:
            text = NUMBERS_RE.sub(" ", text)

        if cfg["remove_punctuation"]:
            text = CLEAN_RE.sub(" ", text)

        if cfg["remove_extra_whitespace"]:
            text = SPACE_RE.sub(" ", text).strip()

        if cfg["lowercase"]:
            text = text.lower()

        if cfg["min_length"] > 0:
            tokens = text.split()
            tokens = [t for t in tokens if len(t) >= cfg["min_length"]]
            text = " ".join(tokens)

        return text

    def get_config(self):
        return self.config