import re


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    value = str(value).lower().strip()
    value = value.replace("&", " and ")
    value = value.replace(".", " ")
    value = re.sub(r"[^a-z0-9\s]", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def normalize_name(value: str) -> str:
    v = normalize_text(value)
    replacements = {
        "private": "private",
        "limited": "limited",
        "ltd": "limited",
        "pvt": "private",
        "corp": "corporation",
        "corporation": "corporation",
        "llp": "llp",
        "inc": "incorporated",
        "incorporated": "incorporated",
    }
    tokens = v.split()
    normalized_tokens = []
    for token in tokens:
        normalized_tokens.append(replacements.get(token, token))
    return " ".join(normalized_tokens)


def normalize_address(value: str) -> str:
    v = normalize_text(value)
    replacements = {
        "st": "street",
        "street": "street",
        "rd": "road",
        "road": "road",
        "ave": "avenue",
        "avenue": "avenue",
        "blvd": "boulevard",
        "boulevard": "boulevard",
        "apt": "apartment",
        "apartment": "apartment",
        "no": "number",
        "number": "number",
    }
    tokens = v.split()
    normalized_tokens = []
    for token in tokens:
        normalized_tokens.append(replacements.get(token, token))
    return " ".join(normalized_tokens)
