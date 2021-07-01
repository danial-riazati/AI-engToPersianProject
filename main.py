import mimetypes
from nltk.stem.snowball import SnowballStemmer
import eng_to_ipa as ipa
from hazm import *
import csv
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

steamer = SnowballStemmer("english")

translationOfIPA = {
    'əʤən': 'یجن',
    'aɪr': 'ایر',
    'aʊr': 'َور',
    'aʊə': 'او',
    'aɪ': 'ای',
    'oʊ': 'و',
    'eɪ': 'ی',
    'aʊ': 'َو',
    'ɜr': 'ِر',
    'ɔɪ': 'وُی',
    'ər': 'ِر',
    'əm': 'ِم',
    'ɪr': 'یر',
    'ɛr': 'ِیر',
    'ɑr': 'ار',
    'ɔr': 'ور',
    'ʊr': 'ور',
    'ks': 'گز',
    'əl': 'ِل',
    'dæ': 'دی',
    'ʤə': 'جو',
    'ən': 'ِن',
    'an': 'ان',
    'hw': 'و',
    'tʃ': 'چ',
    'ʧ': 'چ',
    'dʒ': 'ج',
    'ʤ': 'ج',
    'ʒ': 'ژ',
    'ɪ': 'ی',
    'ɛ': 'ِ',
    'æ': 'َ',
    'ɑ': 'ا',
    'ʌ': 'ا',
    'ə': 'ا',
    'ɔ': 'ا',
    'ʊ': 'و',
    'u': 'و',
    'i': 'ی',
    'ɝ': 'ِر',
    'ɚ': 'ِر',
    'b': 'ب',
    'd': 'د',
    'f': 'ف',
    'ɡ': 'گ',
    'g': 'گ',
    'h': 'ه',
    'k': 'ک',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'ŋ': 'نگ',
    'p': 'پ',
    'r': 'ر',
    's': 'س',
    't': 'ت',
    'v': 'و',
    'w': 'و',
    'x': 'کس',
    'j': 'ی',
    'z': 'ز',
    'θ': 'ت',
    'ð': 'د',
    'ʃ': 'ش',
    ' ': ' ',
    'e': 'ای',
    'c': 'س',
    'o': '',
    'y': 'ی',
    'a': ''
}
constants = {
    'ʧ': 'چ',
    'dʒ': 'ج',
    'ʤ': 'ج',
    'ʒ': 'ژ',
    'ɝ': 'ِر',
    'ɚ': 'ِر',
    'b': 'ب',
    'd': 'د',
    'f': 'ف',
    'ɡ': 'گ',
    'g': 'گ',
    'h': 'ه',
    'k': 'ک',
    'l': 'ل',
    'm': 'م',
    'n': 'ن',
    'ŋ': 'نگ',
    'p': 'پ',
    'r': 'ر',
    's': 'س',
    't': 'ت',
    'v': 'و',
    'w': 'و',
    'x': 'کس',
    'z': 'ز',
    'θ': 'ت',
    'ð': 'د',
    'ʃ': 'ش',
    'y': 'ی'
}
prefixes = ['anti', 'contra', 'counter', 'ante', 'auto', 'semi', 'over', 'post', 'mono', 'under']
dataset = {}
with open("dataset.csv", 'r', encoding='utf-8') as file:
    reader = csv.reader(file)
    dataset = {rows[0]: rows[1] for rows in reader}


def translate(l):
    l = add_aa(l)
    return add_a([translate_word(i) for i in l])


def add_a(l):
    for x in range(len(l)):
        if l[x].startswith("ِ") or l[x].startswith("ُ") or l[x].startswith("َ"):
            l[x] = "ا" + l[x]
    return l


def add_aa(l):
    for x in range(len(l)):
        if l[x].startswith('ɔ') or l[x].startswith('ju') or l[x].startswith('ə'):
            x_list = list(l[x])
            x_list[0] = "آ"
            if l[x].startswith('ju'):
                x_list.pop(1)
            l[x] = "".join(x_list)
        if l[x].startswith('ɪ') or l[x].startswith("oʊ") or l[x].startswith('u'):
            x_list = list(l[x])
            x_list[0] = "ا"
            if l[x].startswith("oʊ"):
                x_list.pop(1)
            l[x] = "".join(x_list)
        if l[x].startswith('i') or l[x].startswith("eɪ"):
            x_list = list(l[x])
            x_list = ['ا'] + x_list
            l[x] = "".join(x_list)
        if l[x].startswith('s'):
            x_list = list(l[x])
            if x_list[1] in constants:
                x_list = ['ا'] + x_list
                l[x] = "".join(x_list)
    return l


def translate_word(data):
    for key in translationOfIPA.keys():
        data = data.replace(key, translationOfIPA[key])
    return data


def prefix_recognizer(s):
    for prefix in prefixes:
        if prefix in s:
            return prefix
    return None


def string_maker(l):
    out = ""
    for x in l:
        out += x + " "
    normalizer = Normalizer()
    return normalizer.normalize("".join(out.rstrip()))


def convert(x):
    tokenized = word_tokenize(x)
    tagged = pos_tag(tokenized)
    if x not in dataset:
        if tagged[0][1] == 'NN' and len(tagged) == 1:
            return x
        steamed = steamer.stem(x)
        if steamed not in dataset:
            return ipa.convert(x)
        else:
            return dataset.get(steamed)
    else:
        return dataset.get(x)


def expression_translate(x):
    converted = convert(x)
    # print(converted)
    converted = converted.replace("ˈ", "")
    converted = converted.replace("ˌ", "")
    converted = converted.replace('*', '')
    ipa_list = converted.split(' ')
    translated_list = translate(ipa_list)
    return string_maker(translated_list)


def eng_to_persian(x):
    prefix = prefix_recognizer(x)
    if prefix is None:
        return expression_translate(x)
    else:
        return expression_translate(prefix) + expression_translate(x.replace(prefix, ''))


file_name = "task2-danialriazati-withoutlabel.csv"
with open(file_name, 'r', encoding="utf8") as file:
    reader = csv.reader(file)
    words = list(reader)
#
for word in words:
    if word[0] != "Id":
        word[1] = eng_to_persian(word[0])

with open(file_name, 'w', newline='', encoding="utf8") as file:
    writer = csv.writer(file)
    for word in words:
        writer.writerow(word)

# for filename in os.listdir('C:\\Users\\dnr\\Desktop\\dataset'):
# if mimetypes.guess_extension(filename) == '.csv':
# with open("dataset.csv", 'r') as file:
#     reader = csv.reader(file)
#     words = list(reader)
#     for word in words:
#         word[1] = ipa.convert(word[0])
#         print(word)
#
#     # for filename in os.listdir('C:\\Users\\dnr\\Desktop\\dataset'):
#     # if mimetypes.guess_extension(filename) == '.csv':
# with open("dataset.csv", 'w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     for word in words:
#         writer.writerow(word)
# tokenized = word_tokenize("book")
# tagged = pos_tag(tokenized)
# print(tagged)
