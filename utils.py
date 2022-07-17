from collections import Counter
import re
import unicodedata
import string
set_punctuations = set(string.punctuation)
list_punctuations_out = ['”', '”', "›", "“", '"', '…', '‘', '’’', "–"]
for e_punc in list_punctuations_out:
    set_punctuations.add(e_punc)
set_punctuations = list(set(set_punctuations))
def norm_text(text):
    text = unicodedata.normalize('NFC', text)
    #text = text.lower()
    text = re.sub(r"òa", "oà", text)
    text = re.sub(r"óa", "oá", text)
    text = re.sub(r"ỏa", "oả", text)
    text = re.sub(r"õa", "oã", text)
    text = re.sub(r"ọa", "oạ", text)
    text = re.sub(r"òe", "oè", text)
    text = re.sub(r"óe", "oé", text)
    text = re.sub(r"ỏe", "oẻ", text)
    text = re.sub(r"õe", "oẽ", text)
    text = re.sub(r"ọe", "oẹ", text)
    text = re.sub(r"ùy", "uỳ", text)
    text = re.sub(r"úy", "uý", text)
    text = re.sub(r"ủy", "uỷ", text)
    text = re.sub(r"ũy", "uỹ", text)
    text = re.sub(r"ụy", "uỵ", text)
    return text


def exception_punctuation(text):
    text = text.replace("\uf0b7", "-")
    text = text.replace("\uf02d", "-")
    text = text.replace("\n", " ")
    return text

def handle_punctuation(text):
    #l_new_char = []
    #for e_char in text:
        #if e_char not in list(set_punctuations):
            #l_new_char.append(e_char)
        #else:
            #l_new_char.append(" {} ".format(e_char))

    #text = "".join(l_new_char)
    for punc in set_punctuations:
        text = text.replace(punc, " " + punc + " ")

    return text


def check_cap(word):
    if len(word) == 1:
        if word.isupper() == True:
            return True
        else:
            return False
    else:
        if word[0].isupper() == True and word[1:].islower() == True:
            return True
        else:
            return False



vocabs_vn_accent = []
with open("vocab.txt", 'r', encoding="utf8") as rf:
    for line in rf.readlines():
        line = line.strip()
        line = norm_text(line)
        # line = remove_accent(line)
        vocabs_vn_accent.append(line)

typo = {"ă": ["aw"], "â": ["aa"], "á": ["as"], "à": ["af"], "ả": ["ar"],
        "ẫ": ["aax"], "ấ": ["aas"], "ầ": ["aaf"], "ẩ": ["aar"], "ậ": ["aaj"],
        "ã": ["ax"], "ạ": ["aj"], "ắ": ["aws"], "ổ": ["oor"], "ỗ": ["oox"],
        "ộ": ["ooj"], "ơ": ["ow"],
        "ằ": ["awf"], "ẳ": ["awr"], "ẵ": ["awx"], "ặ": ["awj"], "ó": ["os"],
        "ò": ["of"], "ỏ": ["or"], "õ": ["ox"], "ọ": ["oj"], "ô": ["oo"],
        "ố": ["oos"], "ồ": ["oof"],
        "ớ": ["ows"], "ờ": ["owf"], "ở": ["owr"], "ỡ": ["owx"], "ợ": ["owj"],
        "é": ["es"], "è": ["ef"], "ẻ": ["er"], "ẽ": ["ex"], "ẹ": ["ej"],
        "ê": ["ee"], "ế": ["ees"], "ề": ["eef"],
        "ể": ["eer"], "ễ": ["eex"], "ệ": ["eej"], "ú": ["us"], "ù": ["uf"],
        "ủ": ["ur"], "ũ": ["ux"], "ụ": ["uj"], "ư": ["uw"], "ứ": ["uws"],
        "ừ": ["uwf"], "ử": ["uwr"], "ữ": ["uwx"],
        "ự": ["uwj"], "í": ["is"], "ì": ["if"], "ỉ": ["ir"], "ị": ["ij"],
        "ĩ": ["ix"], "ý": ["ys"], "ỳ": ["yf"], "ỷ": ["yr"], "ỵ": ["yj"], "ỹ": ["yx"],
        "đ": ["dd"],
        "Ă": ["AW"], "Â": ["AA"], "Á": ["AS"], "À": ["AF"], "Ả": ["AR"],
        "Ẫ": ["AAX"], "Ấ": ["AAS"], "Ầ": ["AAF"], "Ẩ": ["AAR"], "Ậ": ["AAJ"],
        "Ã": ["AX"], "Ạ": ["AJ"], "Ắ": ["AWS"], "Ổ": ["OOR"], "Ỗ": ["OOX"],
        "Ộ": ["OOJ"], "Ơ": ["OW"],
        "Ằ": ["AWF"], "Ẳ": ["AWR"], "Ẵ": ["AWX"], "Ặ": ["AWJ"], "Ó": ["OS"],
        "Ò": ["OF"], "Ỏ": ["OR"], "Õ": ["OX"], "Ọ": ["OJ"], "Ô": ["OO"],
        "Ố": ["OOS"], "Ồ": ["OOF"],
        "Ớ": ["OWS"], "Ờ": ["OWF"], "Ở": ["OWR"], "Ỡ": ["OWX"], "Ợ": ["OWJ"],
        "É": ["ES"], "È": ["EF"], "Ẻ": ["ER"], "Ẽ": ["EX"], "Ẹ": ["EJ"],
        "Ê": ["EE"], "Ế": ["EES"], "Ề": ["EEF"],
        "Ể": ["EER"], "Ễ": ["EEX"], "Ệ": ["EEJ"], "Ú": ["US"], "Ù": ["UF"],
        "Ủ": ["UR"], "Ũ": ["UX"], "Ụ": ["UJ"], "Ư": ["UW"], "Ứ": ["UWS"],
        "Ừ": ["UWF"], "Ử": ["UWR"], "Ữ": ["UWX"],
        "Ự": ["UWJ"], "Í": ["IS"], "Ì": ["IF"], "Ỉ": ["IR"], "Ị": ["IJ"],
        "Ĩ": ["IX"], "Ý": ["YS"], "Ỳ": ["YF"], "Ỷ": ["YR"], "Ỵ": ["YJ"], "Ỹ": ["YX"],
        "Đ": ["DD"]}
special_char = ["a", "e", "o", "d"]

def is_end_of_sentence(i, line):
    exception_list = [
        "Mr.",
        "MR.",
        "GS.",
        "Gs.",
        "PGS.",
        "Pgs.",
        "pgs.",
        "TS.",
        "Ts.",
        "T.",
        "ts.",
        "MRS.",
        "Mrs.",
        "mrs.",
        "Tp.",
        "tp.",
        "Kts.",
        "kts.",
        "BS.",
        "Bs.",
        "Co.",
        "Ths.",
        "MS.",
        "Ms.",
        "TT.",
        "TP.",
        "tp.",
        "ĐH.",
        "Corp.",
        "Dr.",
        "Prof.",
        "BT.",
        "Ltd.",
        "P.",
        "MISS.",
        "miss.",
        "TBT.",
        "Q.",
    ]
    if i == len(line)-1:
        return True
    if line[i+1] != " ":
        return False
    #if i >2 and line[i-1] == ".":
        #retrun
    if line[i-1].isupper() or line[i-2].isupper():
        return False
    if i > 1 and line[i-1] == "." and i < len(line)-2 and line[i+2].islower():
        return False
    #
    # if re.search(r"^(\d+|[A-Za-z])\.", line[:i+1]):
    #     return False
    for w in exception_list:
        pattern = re.compile("%s$" %w)
        if pattern.search(line[:i+1]):
            return False
    return True


def sent_tokenize(line):
    """Do sentence tokenization by using regular expression"""
    sentences = []
    cur_pos = 0
    #line = remove_multi_space(line)
    if not re.search(r"[\.\?!]", line):
        return [line]
    for match in re.finditer(r"[\.\?!]", line):
        _pos = match.start()
        end_pos = match.end()
        if is_end_of_sentence(_pos, line):
            tmpsent = line[cur_pos:end_pos]
            tmpsent = tmpsent.strip()
            cur_pos = end_pos
            sentences.append(tmpsent)
    if len(sentences) == 0:
        sentences.append(line)
    elif cur_pos < len(line)-1:
        sentences.append(line[cur_pos+1:])
    return sentences


def change_to_origin(ch):
    if ch in ['ă', 'â']:
        return "a"
    elif ch == 'đ':
        return 'd'
    elif ch == 'ê':
        return 'e'
    elif ch in ['ô', 'ơ']:
        return 'o'
    elif ch == 'ư':
        return 'u'
    else:
        return ch


def reduce_wrong_word_test(word):
    """ Reduce a word to its original form
    @param word exp: tới
    @return its original form exp: towis
    """
    c = list(word)
    word = "".join(list(set("".join(list(map(lambda x: typo[x][0] if x in typo else x, c))))))
    return word

def check_number_in_word(word):
    for i in range(len(word)):
        if word[i] in ['0','1','2','3','4','5','6','7','8','9']:
            return True
    return False

def check_and_reduce(word):
    if word not in vocabs_vn_accent:
        return reduce_wrong_word_test(word)
    else:
        return word

def check_and_reduce_text(text):
    text = " ".join(list(map(check_and_reduce, text.split())))
    return text


def pad_sents(sents, pad_token):
    """ Pad list of sentences according to the longest sentence in the batch.
    @param sents (list[list[str]]): list of sentences, where each sentence
                                    is represented as a list of words
    @param pad_token (str): padding token
    @returns sents_padded (list[list[str]]): list of sentences where sentences shorter
        than the max length sentence are padded out with the pad_token, such that
        each sentences in the batch now has equal length.
    """

    l = list(map(len, sents))
    max_length = max(l)
    sents_padded = list(map(lambda x: x + [pad_token]*(max_length - len(x)), sents))

    return sents_padded


def read_corpus(file_path):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    """
    data = []
    for line in open(file_path, encoding="utf8"):
        sent = line.strip().split()
        data.append(sent)

    return data

def read_corpus_char(file_path):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    """
    data = []
    for line in open(file_path, encoding="utf8"):
        sent = list(line.strip())
        data.append(sent)

    return data

def exception_punctuation(text):
    text = text.replace("\uf0b7", "-")
    text = text.replace("\n", " ")
    text = text.replace("\a", " ")
    text = text.replace("V/v", "- Về việc")
    return text
    
def read_list_char(path):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    """
    data = []
    for line in path:
        sent = list(line.strip())
        data.append(sent)

    return data

def max_length_char(length_char):
    # [[2, 2, 6, 5], [4, 2, 2, 4]] -> 6
    m = 0
    for lengths in length_char:
        max_lengths = max(lengths)
        m = max_lengths if max_lengths > m else m
    return m


def find_nearest_index(list_error_index, n):
    l = len(list_error_index)
    if l == 0 :
        return -1
    else:
        for i in range(l):
            if list_error_index[i] >=n:
                return i


def find_index(len_line, len_sent, error_index):
    j = 0
    b = len_sent[j]
    c = 0
    ans = {}
    error_index = error_index.copy()
    for i in range(len(len_line)):
        a = len_line[i]
        ans[i] = []
        while(1):
            if a>=b:
                a = a - b
#                 ans[i] += (error_index[j] + c)
                ans[i] += list(map(lambda x : x+c, error_index[j]))
                c = c + b
                try:
                    j+=1
                    b = len_sent[j]
                except:
                    break
            else:
                idx = find_nearest_index(error_index[j],a)
#                 ans[i] += (error_index[j][:idx] + c)
                if idx != None:
                    ans[i] += list(map(lambda x : x+c, error_index[j][:idx]))
    #                 error_index[j] = error_index[j][idx:] - a
                    error_index[j] = list(map(lambda x: x - a, error_index[j][idx:]))
                else:
                    ans[i] += list(map(lambda x: x + c, error_index[j][:]))
                    error_index[j] = []
                b = b - a
                c = 0
                break
    return ans


def parallel_index(text, index):
    ans = []
    text = text.replace("\n", "\a").split()
    for i in range(len(text)):
        text[i] = handle_punctuation(exception_punctuation(text[i]).strip())
        l = len(text[i].split())
        ans += [i]*l
    return [ans[i] for i in index ]
    
def check_sent(sent, v):
    count = 0 
    for word in sent:
        if word.lower() in v and word not in set_punctuations:
            count +=1
            if count > 4 :
                return True
    return False

list_up = ["Độc", "Tập"]
#list_low = ["nếu", "liên"]
exception_error = {"trích":1, "yếu":2, "độ":3, "so":4, "mật":5, "khớp":6, "dấu" : 7 }

def find_true(c, c_upper, s1, s, v):
    ans = []
    for i in range(len(s1)):
        error_index = []
        len_s1_i = len(s1[i])
        if check_sent(s1[i], v) or "nghị quyết" in s[i]:
            if s[i].isupper():
                for j in range(len_s1_i):
                    if c[i][j].item() == 1 and s1[i][j].lower() not in exception_error:
                        error_index.append(j)
            else:
                for j in range(len_s1_i):
                    if c[i][j].item() == 0:
                        #if c_upper[i][j].item() == 1 and s1[i][j].islower() and s1[i][j] not in list_low:
                        if c_upper[i][j].item() == 1 and s1[i][j].islower():
                            error_index.append(j)
                        elif c_upper[i][j].item() == 0 and check_cap(s1[i][j]) and s1[i][j] not in list_up:
                        #elif c_upper[i][j].item() == 0 and check_cap(s1[i][j]):
                            error_index.append(j)
                    else:
                        if not s1[i][j].isupper() and s1[i][j].lower() not in exception_error:
                            error_index.append(j)
        ans.append(error_index)
        error_index = []
    return ans
