import os
import shutil
import sentencepiece as spm
from keyword import *
from token import EXACT_TOKEN_TYPES
from pathlib import Path
from abc import ABCMeta, abstractmethod
from lexer import (
    tokenize,
    generate_tokens,
    TokenInfo,
    tok_name,
    NAME,
    NUMBER,
    STRING,
    NEWLINE,
    NL,
    WHITESPACE,
    ENCODING,
    ENDMARKER,
    COMMENT,
    N_TOKENS,
)
from typing import Generator


MAX_VOCAB_SIZE = 16000
PADDING_TOK = 0
START_TOK = 1
EOF_TOK = 2
UNK_TOK = 3
MASK_TOK = 4
SPECIAL_TOKENS = 5


def _after_sp_training(data_path: str, prefix: str):
    os.makedirs(data_path, exist_ok=True)
    for file in ["model", "vocab"]:
        shutil.move(f"{prefix}.{file}", os.path.join(data_path, f"{prefix}.{file}"))


def _load_sp(data_path: str, prefix: str) -> spm.SentencePieceProcessor:
    model_path = Path(__file__).parent.resolve().joinpath(data_path, f"{prefix}.model")
    return spm.SentencePieceProcessor(model_file=str(model_path))


class Tokenizer(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def get_vocab_size(cls) -> int:
        pass

    @classmethod
    @abstractmethod
    def train(cls, dataset_path: str):
        pass

    @abstractmethod
    def tokenizes(self, program: str) -> list[int]:
        pass

    @abstractmethod
    def tokenize(self, filepath: str) -> list[int]:
        pass

    @abstractmethod
    def untokenize(self, tokens: list[int]) -> str:
        pass


class SpTokenizer(Tokenizer):
    TOKENIZER_DATA_PATH = "tokenizer_data"
    TOKENIZER_PREFIX = "tokenizer"

    def __init__(self) -> None:
        self.sp = _load_sp(self.TOKENIZER_DATA_PATH, self.TOKENIZER_PREFIX)

    @classmethod
    def get_vocab_size(cls) -> int:
        return MAX_VOCAB_SIZE

    @classmethod
    def train(cls, dataset_path: str):
        files = os.listdir(dataset_path)
        input = ",".join(map(lambda file: os.path.join(dataset_path, file), files))
        spm.SentencePieceTrainer.train(
            input=input,
            model_prefix=cls.TOKENIZER_PREFIX,
            vocab_size=MAX_VOCAB_SIZE,
            remove_extra_whitespaces=False,
            add_dummy_prefix=False,
            pad_id=PADDING_TOK,
            bos_id=START_TOK,
            eos_id=EOF_TOK,
            unk_id=UNK_TOK,
            control_symbols=["<mask>"],
            user_defined_symbols=[
                "\n",
                "\t",
                *kwlist,
                *softkwlist,
                *EXACT_TOKEN_TYPES.keys(),
                *[str(i) for i in range(10)],
            ],
        )
        _after_sp_training(cls.TOKENIZER_DATA_PATH, cls.TOKENIZER_PREFIX)

    def tokenizes(self, program: str) -> list[int]:
        return self.sp.encode(program) + [EOF_TOK]

    def tokenize(self, filepath: str) -> list[int]:
        with open(filepath, "r", encoding="UTF-8") as f:
            return self.tokenizes(f.read())

    def untokenize(self, tokens: list[int]) -> str:
        return self.sp.decode(tokens)

    def id_to_token(self, id):
        return self.sp.id_to_piece(id).replace("▁", " ")

    def id_to_token_raw(self, id):
        return self.sp.id_to_piece(id)

    def token_to_id(self, token):
        return self.sp.piece_to_id(token)


class SpNoCustomTokenizer(SpTokenizer):
    TOKENIZER_DATA_PATH = "tokenizer_no_custom_data"
    TOKENIZER_PREFIX = "tokenizer_no_custom"

    @classmethod
    def train(cls, dataset_path: str):
        files = os.listdir(dataset_path)
        input = ",".join(map(lambda file: os.path.join(dataset_path, file), files))
        spm.SentencePieceTrainer.train(
            input=input,
            model_prefix=cls.TOKENIZER_PREFIX,
            vocab_size=MAX_VOCAB_SIZE,
            remove_extra_whitespaces=False,
            pad_id=PADDING_TOK,
            bos_id=START_TOK,
            eos_id=EOF_TOK,
            unk_id=UNK_TOK,
            control_symbols=["<mask>"],
        )
        _after_sp_training(cls.TOKENIZER_DATA_PATH, cls.TOKENIZER_PREFIX)


class AdHocTokenizer(Tokenizer):
    TOKENIZER_DATA_PATH = "ad_hoc_tokenizer_data"
    TOKENIZER_PREFIX = "ad_hoc_tokenizer"
    EXTRACTED_NAMES_FILE = "extracted_names"
    WHITESPACES = " \t\n\r\f"
    OTHER_TOKENS = 31

    def __init__(self) -> None:
        from config import AD_HOC_USE_SP

        self.use_sp = AD_HOC_USE_SP

        cur_token_id = N_TOKENS + SPECIAL_TOKENS
        self.kw_ids = {kw: i + cur_token_id for (i, kw) in enumerate(kwlist)}

        cur_token_id += len(self.kw_ids)
        self.softkw_ids = {
            softkw: i + cur_token_id for (i, softkw) in enumerate(softkwlist)
        }

        cur_token_id += len(self.softkw_ids)
        self.whitespace_ids = {
            ws: i + cur_token_id for (i, ws) in enumerate(self.WHITESPACES)
        }

        cur_token_id += len(self.whitespace_ids)
        self.COMMENT_LEADING_WS = cur_token_id
        self.NUMBER_UNDERSCORE = (cur_token_id := cur_token_id + 1)
        self.NUMBER_HEX = (cur_token_id := cur_token_id + 1)
        self.NUMBER_BIN = (cur_token_id := cur_token_id + 1)
        self.NUMBER_OCT = (cur_token_id := cur_token_id + 1)
        self.NUMBER_EXP = (cur_token_id := cur_token_id + 1)
        self.NUMBER_IMG = (cur_token_id := cur_token_id + 1)
        self.NUMBER_LCASE = (cur_token_id := cur_token_id + 1)
        self.NUMBER_UCASE = (cur_token_id := cur_token_id + 1)
        self.NUMBER_LEADING_POINT = (cur_token_id := cur_token_id + 1)
        self.NUMBER_TRAILING_POINT = (cur_token_id := cur_token_id + 1)
        self.NUMBER_EXP_PLUS = (cur_token_id := cur_token_id + 1)
        self.NUMBER_LEADING_PLUS = (cur_token_id := cur_token_id + 1)
        self.NUMBER_LEADING_ZEROS = (cur_token_id := cur_token_id + 1)
        self.NUMBER_TRAILING_ZEROS = (cur_token_id := cur_token_id + 1)
        self.STRING_SQUOTE = (cur_token_id := cur_token_id + 1)
        self.STRING_DQUOTE = (cur_token_id := cur_token_id + 1)
        self.STRING_TRIPLE_SQUOTE = (cur_token_id := cur_token_id + 1)
        self.STRING_TRIPLE_DQUOTE = (cur_token_id := cur_token_id + 1)
        self.STRING_MULTILINE = (cur_token_id := cur_token_id + 1)
        self.STRING_PREFIX_B = (cur_token_id := cur_token_id + 1)
        self.STRING_PREFIX_R = (cur_token_id := cur_token_id + 1)
        self.STRING_PREFIX_U = (cur_token_id := cur_token_id + 1)
        self.STRING_PREFIX_F = (cur_token_id := cur_token_id + 1)
        self.STRING_PREFIX_LCASE = (cur_token_id := cur_token_id + 1)
        self.STRING_PREFIX_UCASE = (cur_token_id := cur_token_id + 1)
        self.NAME_CAMEL_CASE = (cur_token_id := cur_token_id + 1)
        self.NAME_PASCAL_CASE = (cur_token_id := cur_token_id + 1)
        self.NAME_SNAKE_CASE = (cur_token_id := cur_token_id + 1)
        self.NAME_UPPERCASE = (cur_token_id := cur_token_id + 1)
        self.NAME_MIXED_CASE = (cur_token_id := cur_token_id + 1)

        assert cur_token_id + 1 == self._get_base_token_count()

        self.number_letters = {
            "x": self.NUMBER_HEX,
            "b": self.NUMBER_BIN,
            "o": self.NUMBER_OCT,
            "e": self.NUMBER_EXP,
            "J": self.NUMBER_IMG,
        }

        self.string_prefixes = {
            "b": self.STRING_PREFIX_B,
            "r": self.STRING_PREFIX_R,
            "u": self.STRING_PREFIX_U,
            "f": self.STRING_PREFIX_F,
        }

        if self.use_sp:
            self.sp = _load_sp(self.TOKENIZER_DATA_PATH, self.TOKENIZER_PREFIX)

    @classmethod
    def get_vocab_size(cls) -> int:
        from config import AD_HOC_USE_SP

        return MAX_VOCAB_SIZE if AD_HOC_USE_SP else cls._get_base_token_count()

    @classmethod
    def _get_base_token_count(cls) -> int:
        return (
            SPECIAL_TOKENS
            + N_TOKENS
            + len(kwlist)
            + len(softkwlist)
            + len(cls.WHITESPACES)
            + cls.OTHER_TOKENS
        )

    @classmethod
    def train(cls, dataset_path: str):
        from config import AD_HOC_USE_SP

        if not AD_HOC_USE_SP:
            return

        names_file = cls._extract_names(dataset_path)
        vocab_size = MAX_VOCAB_SIZE - cls._get_base_token_count() - SPECIAL_TOKENS
        spm.SentencePieceTrainer.train(
            input=names_file,
            model_prefix=cls.TOKENIZER_PREFIX,
            vocab_size=vocab_size,
            remove_extra_whitespaces=False,
            add_dummy_prefix=False,
            pad_id=PADDING_TOK,
            bos_id=START_TOK,
            eos_id=EOF_TOK,
            unk_id=UNK_TOK,
            control_symbols=["<mask>"],
        )
        _after_sp_training(cls.TOKENIZER_DATA_PATH, cls.TOKENIZER_PREFIX)

    @classmethod
    def _extract_names(cls, dataset_path: str) -> str:
        names_file = os.path.join(cls.TOKENIZER_DATA_PATH, cls.EXTRACTED_NAMES_FILE)
        if os.path.exists(names_file):
            return names_file

        names_set = set()
        kwset = set(kwlist)
        softkwset = set(softkwlist)
        for filepath in os.listdir(dataset_path):
            with open(os.path.join(dataset_path, filepath), "rb") as f:
                try:
                    for token in tokenize(f.readline):
                        if (
                            token.type == NAME
                            and not token.string in kwset
                            and not token.string in softkwset
                        ):
                            names_set.add(token.string)
                except Exception:
                    pass

        os.makedirs(cls.TOKENIZER_DATA_PATH, exist_ok=True)
        with open(names_file, "w", encoding="UTF-8") as f:
            f.write("\n".join(names_set))

        return names_file

    def tokenizes(self, program: str) -> list[int]:
        lines = program.splitlines(True)
        generator = generate_tokens(iter(lines).__next__)
        return self._from_generator(generator)

    def tokenize(self, filepath: str) -> list[int]:
        with open(filepath, "rb") as f:
            return self._from_generator(tokenize(f.readline))

    def _from_generator(self, gen: Generator[TokenInfo, None, None]) -> list[int]:
        base_token_count = self._get_base_token_count()
        tokens = []
        for token in gen:
            if token.type == ENCODING or token.type == ENDMARKER:
                continue

            elif token.type in [WHITESPACE, NEWLINE, NL]:
                for char in token.string:
                    tokens.append(self.whitespace_ids[char])
                continue

            elif token.type == COMMENT:
                tokens.append(token.exact_type + SPECIAL_TOKENS)
                if len(token.string) > 1 and token.string[1] in " \t":
                    tokens.append(self.COMMENT_LEADING_WS)
                continue

            elif token.type == NUMBER:
                tokens.append(token.exact_type + SPECIAL_TOKENS)

                if token.string[0] == ".":
                    tokens.append(self.NUMBER_LEADING_POINT)
                elif token.string[0] == "+":
                    tokens.append(self.NUMBER_LEADING_PLUS)
                elif token.string[0] == "0":
                    for char in token.string[1:]:
                        if char != "0":
                            if char.isdigit():
                                tokens.append(self.NUMBER_LEADING_ZEROS)
                            break

                if "_" in token.string:
                    tokens.append(self.NUMBER_UNDERSCORE)

                for letter, id in self.number_letters.items():
                    if letter in token.string:
                        tokens.append(id)
                        tokens.append(self.NUMBER_LCASE)
                    elif letter.upper() in token.string:
                        tokens.append(id)
                        tokens.append(self.NUMBER_UCASE)

                if "e+" in token.string or "E+" in token.string:
                    tokens.append(self.NUMBER_EXP_PLUS)

                if token.string[-1] == ".":
                    tokens.append(self.NUMBER_TRAILING_POINT)
                elif "." in token.string:
                    start = token.string.find("e") != -1
                    if start == -1:
                        start = token.string.find("E") != -1
                    if start == -1:
                        start = len(token.string) - 1

                    if token.string[start] == "0":
                        for i in range(start, 0, -1):
                            char = token.string[i]
                            if char != "0":
                                if char.isdigit():
                                    tokens.append(self.NUMBER_TRAILING_ZEROS)
                                break

                continue

            elif token.type == STRING:
                tokens.append(token.exact_type + SPECIAL_TOKENS)

                prefix = []
                for char in token.string:
                    if char == "'" or char == '"':
                        break
                    prefix.append(char)

                for char in prefix:
                    lower = char.lower()
                    tokens.append(self.string_prefixes[lower])
                    tokens.append(
                        self.STRING_PREFIX_LCASE
                        if lower == char
                        else self.STRING_PREFIX_UCASE
                    )

                quote_start = len(prefix)
                if token.string[quote_start] == '"':
                    tokens.append(
                        self.STRING_TRIPLE_DQUOTE
                        if token.string[quote_start + 1] == '"'
                        else self.STRING_DQUOTE
                    )
                else:
                    tokens.append(
                        self.STRING_TRIPLE_SQUOTE
                        if token.string[quote_start + 1] == "'"
                        else self.STRING_SQUOTE
                    )

                if "\n" in token.string:
                    tokens.append(self.STRING_MULTILINE)

                continue

            elif token.type == NAME:
                if iskeyword(token.string):
                    tokens.append(self.kw_ids[token.string])
                elif issoftkeyword(token.string):
                    tokens.append(self.softkw_ids[token.string])
                else:
                    tokens.append(token.exact_type + SPECIAL_TOKENS)

                    has_lower = False
                    has_upper = False
                    has_underscore = False
                    for char in token.string:
                        if char == "_":
                            has_underscore = True
                        elif char.isupper():
                            has_upper = True
                        elif char.islower():
                            has_lower = True

                    if has_lower and has_upper and not has_underscore:
                        tokens.append(
                            self.NAME_PASCAL_CASE
                            if token.string[0].isupper()
                            else self.NAME_CAMEL_CASE
                        )
                    elif has_underscore and not has_upper:
                        tokens.append(self.NAME_SNAKE_CASE)
                    elif has_upper and not has_lower:
                        tokens.append(self.NAME_UPPERCASE)
                    elif has_upper and has_lower and has_underscore:
                        tokens.append(self.NAME_MIXED_CASE)

                    if self.use_sp:
                        tokens.extend(
                            tok - SPECIAL_TOKENS + base_token_count
                            for tok in self.sp.encode(token.string)
                        )

                continue

            tokens.append(token.exact_type + SPECIAL_TOKENS)

        tokens.append(EOF_TOK)
        return tokens

    def untokenize(self, tokens: list[int]) -> str:
        base_token_count = self._get_base_token_count()
        untokenized = []
        inv_kw_ids = {v: k for k, v in self.kw_ids.items()}
        max_kw_id = max(inv_kw_ids)
        inv_softkw_ids = {v: k for k, v in self.softkw_ids.items()}
        max_softkw_id = max(inv_softkw_ids)
        inv_whitespace_ids = {v: k for k, v in self.whitespace_ids.items()}
        max_whitespace_id = max(inv_whitespace_ids)
        special_tok_name = {
            value: name
            for name, value in globals().items()
            if isinstance(value, int) and not name.startswith("_")
        }
        other_tok_name = {
            value: name
            for name, value in vars(self).items()
            if isinstance(value, int) and not name.startswith("_")
        }

        for token in tokens:
            if token < SPECIAL_TOKENS:
                untokenized.append(special_tok_name[token])
            elif token < N_TOKENS + SPECIAL_TOKENS:
                untokenized.append(tok_name[token - SPECIAL_TOKENS])
            elif token <= max_kw_id:
                untokenized.append(repr(inv_kw_ids[token]))
            elif token <= max_softkw_id:
                untokenized.append(repr(inv_softkw_ids[token]))
            elif token <= max_whitespace_id:
                untokenized.append(repr(inv_whitespace_ids[token]))
            elif token <= max_whitespace_id + self.OTHER_TOKENS:
                untokenized.append(other_tok_name[token])
            elif self.use_sp:
                untokenized.append(
                    self.sp.decode(token + SPECIAL_TOKENS - base_token_count)
                )

        return " ".join(untokenized)


if __name__ == "__main__":
    from config import TOKENIZER

    TOKENIZER.train(os.path.join(os.pardir, "Crawler", "python_dataset"))
