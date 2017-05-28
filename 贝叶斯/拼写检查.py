import re
from collections import Counter
 
def words(text): return re.findall(r'\w+', text.lower())    #文章拆分成单词存入列表

WORDS = Counter(words(open('big.txt').read()))  #记录每个单词个数
#print(open('big.txt').read())
#print(type(open('big.txt').read()))
#print(words(open('big.txt').read()))
#print(WORDS)

def P(word,N=sum(WORDS.values())):
    return WORDS[word]/N   #计算单词的概率

#编辑距离 增 删 改 
def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)  #返回已知词
def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))



def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)



if __name__ == '__main__':
    word = ('banan')
    print (correction(word))
    #print(WORDS['begin'],WORDS['banana'])

