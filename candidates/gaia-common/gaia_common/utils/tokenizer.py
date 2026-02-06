import tiktoken

try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = tiktoken.get_encoding("gpt2")

def count_tokens(text: str) -> int:
    """
    Counts the number of tokens in a given text string using tiktoken.
    """
    if not isinstance(text, str):
        return 0
    return len(encoding.encode(text))
