import re

_PERSIAN_ARABIC_DIGITS = '۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩'
_LATIN_DIGITS = '01234567890123456789'
_DIGIT_TRANSLATION = str.maketrans(_PERSIAN_ARABIC_DIGITS, _LATIN_DIGITS)

_LETTER_MAP = {
    'ي': 'ی',
    'ك': 'ک',
    'ة': 'ه',
    'أ': 'ا',
    'إ': 'ا',
    'ؤ': 'و',
}


def normalize(text):
    text = text.translate(_DIGIT_TRANSLATION)

    for source, target in _LETTER_MAP.items():
        text = text.replace(source, target)

    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r' *\n *', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()
