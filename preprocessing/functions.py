import re

unicode_stoplist = [0x2002,0x2005,0x2009,0x200a]
replacements = {u:'' for u in unicode_stoplist}

class stanza_word:
    def __init__(self,ix,text,head,deprel,span):
        self.id = ix
        self.text = text
        self.head = head 
        self.deprel = deprel
        self.span = span
    
    def __repr__(self):
        return '{\n' + f"  id:{self.id}\n" + f"  text:{self.text}\n" + f"  head:{self.head}\n" + \
                f"  deprel:{self.deprel}\n" + f"  span:{self.span}\n" + '}'

##### deprecated: treat all compounds like A-B (contains only one '-') by seperating it into three words 'A', '-' and 'B'.
"""def re_organise_words(words,stanza_parser_pretokenized,subj_start,subj_end,obj_start,obj_end,
                      subj_obj_start,subj_obj_end,lst_special_symbols=['-','/']):
    # reorganise stanza word objects, cuz misc is a somehow strange usage.
    # treat the problem that some words will not be seperated by stanza during tokenization
    # e.g. "imatinib-resistant" will be considered as a single token.
    res = []
    # map old word indexes to new ones 
    remap = {} 
    new_created_indexes = []
    offset = 0
    for word in words:
        word_text = word.text
        word_start, word_end = decode_misc(word.misc)
        for ss in lst_special_symbols:
            if ss in word_text:
                tmp = word_text.split(ss)
                if len(tmp) != 2:
                    res.append(stanza_word(word.id+offset,word.text,word.head,word.deprel,
                                          *decode_misc(word.misc)))
                    remap[word.id] = word.id+offset
                    print(f"special token: {word_text}")
                    break
                curr_id = word.id + offset
                curr_start, curr_end = decode_misc(word.misc)
                tmp_compound = []
                #print(tmp)
                if len(tmp[0]) > 0:
                    #print(curr_start,len(tmp[0]),tmp[0])
                    res.append(stanza_word(curr_id,tmp[0].strip(),word.head,word.deprel,
                                                         curr_start,curr_start+len(tmp[0].strip())))
                    tmp_compound.append(tmp[0].strip())
                    curr_start += len(tmp[0])
                res.append(stanza_word(curr_id+len(tmp_compound),ss,word.head,word.deprel,
                                                      curr_start,curr_start+len(ss)))
                tmp_compound.append(ss)
                curr_start += len(ss)
                if len(tmp[1]) > 0:
                    res.append(stanza_word(curr_id+len(tmp_compound),tmp[1].strip(),word.head,word.deprel,
                                                          curr_start,curr_start+len(tmp[1].strip())))
                    tmp_compound.append(tmp[1].strip())
                # do a "little" syntactic analysis on this word containing special tokens like '-' 
                tmp_doc = stanza_parser_pretokenized([tmp_compound])
                tmp_words = tmp_doc.sentences[0].words
                # link the root of compound to the head of compound, other tokens of compound will be 
                # linked to tokens within the compound according to the syntactic analysis
                pseudo_ix = {0:word.head,**{i:curr_id+i-1 for i in range(1,4)}}
                head_ix = [w.head for w in tmp_words]
                head_deprels = [w.deprel for w in tmp_words]
                for hi, hd, re_w in zip(head_ix,head_deprels,res[-len(head_ix):]):
                    re_w.head = pseudo_ix[hi]
                    if hd != "root":
                        re_w.deprel = hd
                offset += len(head_ix) - 1
                # the root of compound will serve as the "representative" of compound
                remap[word.id] = curr_id + head_deprels.index("root")
                for i, hd in enumerate(head_deprels):
                    if hd != "root":
                        new_created_indexes.append(curr_id+i)
                break
        else:
            res.append(stanza_word(word.id+offset,word.text,word.head,word.deprel,
                                          *decode_misc(word.misc)))
            remap[word.id] = word.id+offset
    # update syntactic head indexes of words
    for re_w in res:
        if re_w.id not in new_created_indexes:
            if re_w.head != 0:
                re_w.head = remap[re_w.head]
    return res"""

#=== MODIFICATION ===
# I thought to seperate ALL compounds containing only one symbol like '-', but compounds seem to be 
# much more complicated than expected in the real dataset. Besides, it seems that compounds 
# in the form of NOUN-ADJ rarely exist or stanza fail to predict the pos-tag as ADJ or dependency as "amod"
# even in case of existence (due to that NOUN is unseen in the training set I presume)
# Therefore, I turn to seperate compounds only when they contain targeted arguments.
#=== FOR MEMORY
# 1) why seperate these compound tokens containing punctuations like '-'?
# : sometimes targeted argument is within compounds, e.g. $$ COX-2 $$-mediated
# 2) why not retokenize full sentence with these compounds segmented?
# : from the results of several tests, trait 'A-B' as a single token seems to be a convention,
#   if seperate 'A-B' like tokens manually and refeed them to the parser, the parser gives
#   results somehow false or 'dep' dependencies.
#====================
def re_organise_words_old(words,stanza_parser_pretokenized,subj_ix,obj_ix,subj_obj_ix,
                      lst_special_symbols=['-','/','+']):
    # reorganise stanza word objects, cuz misc is a somehow strange usage.
    # treat the problem that some words will not be seperated by stanza during tokenization
    # e.g. "imatinib-resistant" will be considered as a single token.
    subj_exists = obj_exists = subj_obj_exists = False
    if len(subj_ix) > 0:
        subj_start, subj_end = subj_ix
        subj_exists = True
    if len(obj_ix) > 0:
        obj_start, obj_end = obj_ix
        obj_exists = True
    if len(subj_obj_ix) > 0:
        subj_obj_start,subj_obj_end = subj_obj_ix
        subj_obj_exists = True
    res = []
    # map old word indexes to new ones 
    remap = {} 
    new_created_indexes = []
    offset = 0
    for word in words:
        word_text = word["text"]
        word_start, word_end = word["span"]
        curr_id = word["id"] + offset
        split_ix = []
        #print(obj_start,obj_end,word_start,word_end)
        if subj_exists and ((subj_start >= word_start and subj_end < word_end) or \
            (subj_start > word_start and subj_end == word_end)):
            split_ix.append(subj_start-word_start)
            split_ix.append(subj_end-word_start)
            # in case argument overlaps over two words 
        elif subj_exists and (word_start < subj_start < word_end and subj_end >= word_end):
            split_ix.append(subj_start-word_start)
        elif subj_exists and (subj_start < word_start and word_start < subj_end <= word_end):
            split_ix.append(subj_end-word_start)

        if obj_exists and ((obj_start >= word_start and obj_end < word_end) or \
            (obj_start > word_start and obj_end == word_end)):
            split_ix.append(obj_start-word_start)
            split_ix.append(obj_end-word_start)
        elif obj_exists and (word_start < obj_start < word_end and obj_end >= word_end):
            split_ix.append(obj_start-word_start)
        elif obj_exists and (obj_start < word_start and word_start < obj_end <= word_end):
            split_ix.append(obj_end-word_start)

        if subj_obj_exists and ((subj_obj_start >= word_start and subj_obj_end < word_end) or \
            (subj_obj_start > word_start and subj_obj_end == word_end)):
            split_ix.append(subj_obj_start-word_start)
            split_ix.append(subj_obj_end-word_start)
        elif subj_obj_exists and (word_start < subj_obj_start < word_end and subj_obj_end >= word_end):
            split_ix.append(subj_obj_start-word_start)
        elif subj_obj_exists and (subj_obj_start < word_start and word_start < subj_obj_end <= word_end):
            split_ix.append(subj_obj_end-word_start)

        # if a part of a word is tagged as argument, it is considered as a compound
        #print(word_start,word_end)
        print(subj_start,subj_end,word_start,word_end,split_ix)
        #assert len(split_ix) % 2 == 0, "the number of splitting points of compound MUST BE even."
        if len(split_ix) > 0:
            tmp_compound, tmp_span = [], []
            curr_tmp_ix = 0
            #print(word_start,word_end,word_text)
            #print(split_ix)
            split_ix = sorted(list(set(split_ix + [word_end-word_start]))) # in case subject or object end equals word end
            for ix in split_ix:
                #print(ix,curr_tmp_ix)
                if ix > curr_tmp_ix: 
                    tmp_compound.append(word_text[curr_tmp_ix:ix].strip())
                    tmp_span.append((word_start+curr_tmp_ix,word_start+ix))
                    curr_tmp_ix = ix
            #print(tmp_compound)
            #print(tmp_span)
            # do a "little" syntactic analysis on this word containing special tokens like '-' 
            tmp_words = stanza_parser_pretokenized([tmp_compound]).sentences[0].words
            pseudo_ix = {0:word["head"],**{i:curr_id+i-1 for i in range(1,4)}}
            head_ix = [w.head for w in tmp_words]
            # link the root of compound to the head of compound, other tokens of compound will be 
            # linked to tokens within the compound according to the syntactic analysis
            for i, w in enumerate(tmp_words):
                if w.head != 0:
                    res.append(stanza_word(curr_id+i,tmp_compound[i],pseudo_ix[w.head],w.deprel,
                                           tmp_span[i]))
                    new_created_indexes.append(curr_id+i)
                else:
                    res.append(stanza_word(curr_id+i,tmp_compound[i],pseudo_ix[w.head],word["deprel"],
                                           tmp_span[i]))
                    remap[word["id"]] = curr_id + i
            offset += len(head_ix) - 1
        else:
            res.append(stanza_word(word["id"]+offset,word["text"],word["head"],word["deprel"],word["span"]))
            remap[word["id"]] = word["id"]+offset
    # update syntactic head indexes of words
    for re_w in res:
        if re_w.id not in new_created_indexes and re_w.head != 0:
            re_w.head = remap[re_w.head]
    return res

def re_organise_words(cleaned_sent,words,stanza_parser_pretokenized,subj_ix,obj_ix,subj_obj_ix,
                      lst_special_symbols=['-','/','+']):
    # reorganise stanza word objects, cuz misc is a somehow strange usage.
    # treat the problem that some words will not be seperated by stanza during tokenization
    # e.g. "imatinib-resistant" will be considered as a single token.
    subj_exists = obj_exists = subj_obj_exists = False
    if len(subj_ix) > 0:
        subj_start, subj_end = subj_ix
        subj_exists = True
    if len(obj_ix) > 0:
        obj_start, obj_end = obj_ix
        obj_exists = True
    if len(subj_obj_ix) > 0:
        subj_obj_start,subj_obj_end = subj_obj_ix
        subj_obj_exists = True
    res = []
    # map old word indexes to new ones 
    remap = {} 
    new_created_indexes = []
    offset = 0
    for word in words:
        word_text = word["text"]
        word_start, word_end = word["span"]
        curr_id = word["id"] + offset
        split_ix = []
        #print(obj_start,obj_end,word_start,word_end)
        if subj_exists and ((subj_start >= word_start and subj_end < word_end) or \
            (subj_start > word_start and subj_end == word_end)):
            split_ix.append(subj_start)
            split_ix.append(subj_end)
            # in case argument overlaps over two words 
        elif subj_exists and (word_start < subj_start < word_end and subj_end >= word_end):
            split_ix.append(subj_start)
        elif subj_exists and (subj_start < word_start and word_start < subj_end <= word_end):
            split_ix.append(subj_end)

        if obj_exists and ((obj_start >= word_start and obj_end < word_end) or \
            (obj_start > word_start and obj_end == word_end)):
            split_ix.append(obj_start)
            split_ix.append(obj_end)
        elif obj_exists and (word_start < obj_start < word_end and obj_end >= word_end):
            split_ix.append(obj_start)
        elif obj_exists and (obj_start < word_start and word_start < obj_end <= word_end):
            split_ix.append(obj_end)

        if subj_obj_exists and ((subj_obj_start >= word_start and subj_obj_end < word_end) or \
            (subj_obj_start > word_start and subj_obj_end == word_end)):
            split_ix.append(subj_obj_start)
            split_ix.append(subj_obj_end)
        elif subj_obj_exists and (word_start < subj_obj_start < word_end and subj_obj_end >= word_end):
            split_ix.append(subj_obj_start)
        elif subj_obj_exists and (subj_obj_start < word_start and word_start < subj_obj_end <= word_end):
            split_ix.append(subj_obj_end)

        # if a part of a word is tagged as argument, it is considered as a compound
        #assert len(split_ix) % 2 == 0, "the number of splitting points of compound MUST BE even."
        if len(split_ix) > 0:
            """print(word_start,word_end)
            print(subj_start,subj_end,obj_start,obj_end)
            print(word_start,word_end,split_ix)"""
            tmp_compound, tmp_span = [], []
            curr_tmp_ix = word_start
            #print(word_start,word_end,word_text)
            #print(split_ix)
            split_ix = sorted(list(set(split_ix + [word_end]))) # in case subject or object end equals word end
            for ix in split_ix:
                #print(ix,curr_tmp_ix)
                tmp_token = cleaned_sent[curr_tmp_ix:ix].strip()
                if ix > curr_tmp_ix and len(tmp_token) > 0: 
                    tmp_compound.append(tmp_token)
                    tmp_span.append((curr_tmp_ix,ix))
                    curr_tmp_ix = ix
            #print(tmp_compound)
            #print(tmp_span)
            # do a "little" syntactic analysis on this word containing special tokens like '-' 
            tmp_words = stanza_parser_pretokenized([tmp_compound]).sentences[0].words
            pseudo_ix = {0:word["head"],**{i:curr_id+i-1 for i in range(1,4)}}
            head_ix = [w.head for w in tmp_words]
            # link the root of compound to the head of compound, other tokens of compound will be 
            # linked to tokens within the compound according to the syntactic analysis
            for i, w in enumerate(tmp_words):
                if w.head != 0:
                    res.append(stanza_word(curr_id+i,tmp_compound[i],pseudo_ix[w.head],w.deprel,
                                           tmp_span[i]))
                    new_created_indexes.append(curr_id+i)
                else:
                    res.append(stanza_word(curr_id+i,tmp_compound[i],pseudo_ix[w.head],word["deprel"],
                                           tmp_span[i]))
                    remap[word["id"]] = curr_id + i
            offset += len(head_ix) - 1
        else:
            res.append(stanza_word(word["id"]+offset,word["text"],word["head"],word["deprel"],word["span"]))
            remap[word["id"]] = word["id"]+offset
    # update syntactic head indexes of words
    for re_w in res:
        if re_w.id not in new_created_indexes and re_w.head != 0:
            re_w.head = remap[re_w.head]
    return res

def index_arguments(sentence):
    # position arguments:
    # markers are always added like '@@ ARGUMENT @@ NEXT_WORD'
    # subj_ix marks the start of span of the first word in ARGUMENT and 
    # the end of the last word in ARGUMENT (if ARGUMENT is a subject) such that word[subj_ix[0]:subj_ix[1]] gives the full text of the subject
    # subj_ix, obj_ix, subj_obj_ix are tuples (start_ix,end_ix)
    prev_c = ''
    num = 0
    subj_ix, obj_ix, subj_obj_ix = [], [], []
    subj_found = obj_found = subj_obj_found = False
    for i, c in enumerate(sentence):
        if prev_c == '@' and c == '@':
            if not subj_found:
                num += 1
                subj_ix.append(i+2-num*2)
                subj_found = True
            else:
                subj_ix.append(i-2-num*2)
                num += 1
        elif prev_c == '$' and c == '$':
            if not obj_found:
                num += 1
                obj_ix.append(i+2-num*2)
                obj_found = True
            else:
                obj_ix.append(i-2-num*2)
                num += 1
        elif prev_c == '¢' and c == '¢':
            if not subj_obj_found:
                num += 1
                subj_obj_ix.append(i+2-num*2)
                subj_obj_found = True
            else:
                subj_obj_ix.append(i-2-num*2)
                num += 1
        prev_c = c
    return subj_ix, obj_ix, subj_obj_ix

def decode_misc(s,reg_start=r"(?<=\=)(.*)\|",reg_end="(?<=end_char\=)(.*)$"):
    return int(re.findall(reg_start,s)[0]), int(re.findall(reg_end,s)[0])

def update_deps(deps,m):
    # m: word index in the sentence NO_MARKER -> MARKER_ADDED
    updated_deps = []
    for i1, i2, d in deps:
        updated_deps.append((m[i1],m[i2],d))
    return updated_deps

def insert_marker(tokens,deps,start,end,marker_text):
    assert len(tokens) == len(deps), "number of tokens and dependencies not equal."
    token_map = {0:0}
    for i, t in enumerate(tokens):
        if i < start:
            token_map[i+1] = i + 1
        elif start <= i < end:
            token_map[i+1] = i + 3
        else:
            token_map[i+1] = i + 5
    
    deps = update_deps(deps,token_map)
    tokens = tokens[:start] + [marker_text] * 2 + tokens[start:end] +\
                [marker_text] * 2 + tokens[end:]
    summary = {"tokens":tokens,"dependencies":deps}
    if marker_text == '@':
        summary["subj"] = [start,start+1,end+2,end+3]
    if marker_text == '$':
        summary["obj"] = [start,start+1,end+2,end+3]
    if marker_text == '¢':
        summary["subj-obj"] = [start,start+1,end+2,end+3]
    return summary

def insert_marker_of_two_arguments(tokens,deps,start1,end1,start2,end2,marker1,marker2):
    # 1: subject; 2: object;
    assert len(tokens) == len(deps), "number of tokens and dependencies not equal."
    token_map = {0:0}
    curr_num_marker = 0
    indexes = sorted([start1,end1,start2,end2],reverse=True)
    
    for i, t in enumerate(tokens):
        while len(indexes) > 0 and i == indexes[-1]: # use while to handle extreme case #1
            curr_num_marker += 1
            indexes.pop()
        token_map[i+1] = i + 1 + curr_num_marker * 2
    deps = update_deps(deps,token_map)
    #print(len(tokens))
    #print(token_map)
    
    if end1 <= start2:
        tokens = tokens[:start1] + ['@'] * 2 + tokens[start1:end1] + ['@'] * 2 + tokens[end1:start2] + \
                    ['$'] * 2 + tokens[start2:end2] + ['$'] * 2 + tokens[end2:]
        summary = {"tokens":tokens,"dependencies":deps,"subj":[start1,start1+1,end1+2,end1+3],
                   "obj":[start2+4,start2+5,end2+6,end2+7]}
    elif end2 <= start1:
        tokens = tokens[:start2] + ['$'] * 2 + tokens[start2:end2] + ['$'] * 2 + tokens[end2:start1] + \
                    ['@'] * 2 + tokens[start1:end1] + ['@'] * 2 + tokens[end1:]
        summary = {"tokens":tokens,"dependencies":deps,"obj":[start2,start2+1,end2+2,end2+3],
                   "subj":[start1+4,start1+5,end1+6,end1+7]}
    else:
        raise ValueError("impossible subject and object span")
    return summary

#=========== extreme case studies
#=========== 1. end1 equals start2 like in "... ARGUMENT-1 $$ @@ ARGUMENT-2"
#=========== 2. stanza takes compounds in a format like A-B-...-C as a single token during tokenization but 
#               part of compound e.g. A,C, A-B is wrapped by entity markers

def parse_sentence(sentence,stanza_words,stanza_parser_pretokenized):
    # ===== overview of algorithm ======
    # the processing is not complicated but a little tedious.
    # to tangle the problem of parsing sentences with entity markers, this function implements:
    # 1) parsing by using the sentence with entity markers removed (it is pre-assured that all sentences 
    # contain, if there are, only 4 entity markers, i.e. only at the start and end of target arguments, 
    # nowhere else. it is important to ensure that entity markers are not mixed with normal tokens.)
    # 2) get tokens from parsing results and add markers directly to the token sequence 
    # (e.g."@@ Michael @@ lives in $$ LA $$." -> " Michael  lives in  LA ." -> 
    #       ["Michael","lives","in","LA","."] -> 
    #       ["@","@","Michael","@","@","lives","in","$","$","LA","$","$","."])
    # 3) get dependencies as a list of tuples (word_index,head_index,dependency_tag) and remap word indexes 
    #    to their indexes in the new token sequence, this can be directly used if we want to extract word 
    #    embeddings excluding those of entity markers. If we want to add manual dependencies between words 
    #    and entity markers, it is also easy to operate.
    # ==================================
    # find the text span of subject and object by iterating character sequence
    subj_ix, obj_ix, subj_obj_ix = index_arguments(sentence)
    cleaned_sentence = sentence.replace("@@",'').replace("$$",'').replace("¢¢",'')
    subj_start = subj_end = obj_start = obj_end = subj_obj_start = subj_obj_end = -1
    
    # match 
    if len(subj_ix) == 2:
        subj_start, subj_end = subj_ix
    elif len(subj_ix) != 0:
        raise ValueError('impossible subject index')
    if len(obj_ix) == 2:
        obj_start, obj_end = obj_ix
    elif len(obj_ix) != 0:
        raise ValueError('impossible object index')
    if len(subj_obj_ix) == 2:
        subj_obj_start, subj_obj_end = subj_obj_ix
    elif len(subj_obj_ix) != 0:
        raise ValueError('impossible subject-object index')

    # token_start: the first token index of argument
    # token_end: the last token index of argument + 1
    # use [token_start:token_end] should obtain the full argument
    subj_token_start = subj_token_end = obj_token_start = obj_token_end = \
    subj_obj_token_start = subj_obj_token_end = -1
    
    words = re_organise_words(cleaned_sentence,stanza_words,stanza_parser_pretokenized,subj_ix,obj_ix,subj_obj_ix)
    word_ids = [w.id for w in words]
    assert len(set(word_ids)) == len(words) and max(word_ids) == len(words), \
           "error during retokenization:word indexes are NOT consecutive!"
    
    tokens = []
    deps = []
    #print(words)
    for w in words:
        word_start, word_end = w.span
        if word_start == subj_start:
            subj_token_start = w.id - 1 
        if word_end == subj_end:
            subj_token_end = w.id 
        if word_start == obj_start:
            obj_token_start = w.id - 1
        if word_end == obj_end:
            obj_token_end = w.id
        if word_start == subj_obj_start:
            subj_obj_token_start = w.id - 1
        if word_end == subj_obj_end:
            subj_obj_token_end = w.id
        tokens.append(w.text)
        deps.append((w.id,w.head,w.deprel))
    if subj_token_start == subj_token_end == subj_token_start == subj_token_end == obj_token_start == \
        obj_token_end == -1:
        raise ValueError("no argument found.")
    
    #print([w.text for w in words])
    #print(subj_start,subj_end,obj_start,obj_end,subj_obj_start,subj_obj_end)
    #print(subj_token_start,subj_token_end,obj_token_start,obj_token_end,subj_obj_token_start,subj_obj_token_end)
    assert (subj_obj_token_start != -1 and subj_obj_token_end != -1) or \
            (subj_obj_token_start == -1 and subj_obj_token_end == -1), \
            "only start or end span is found. (subj-obj)"
    assert (subj_token_start != -1 and subj_token_end != -1) or \
            (subj_token_start == -1 and subj_token_end == -1), "only start or end span is found. (subj)"
    assert (obj_token_start != -1 and obj_token_end != -1) or \
            (obj_token_start == -1 and obj_token_end == -1), "only start or end span is found. (obj)"
    # cases where only one target argument
    if subj_obj_token_start != -1:
        sent_doc = insert_marker(tokens,deps,subj_obj_token_start,subj_obj_token_end,'¢')
        return sent_doc, parse_legitimacy_check(sentence,words,sent_doc)
    if subj_token_start != -1 and obj_token_start == -1:
        sent_doc = insert_marker(tokens,deps,subj_token_start,subj_token_end,'@')
        return sent_doc, parse_legitimacy_check(sentence,words,sent_doc)
    if obj_token_start != -1 and subj_token_start == -1:
        sent_doc = insert_marker(tokens,deps,obj_token_start,obj_token_end,'$')
        return sent_doc, parse_legitimacy_check(sentence,words,sent_doc)
    assert subj_token_start != -1 and subj_token_end != -1 and obj_token_start != -1 and \
            obj_token_end != -1, "subject or object span detection error."
    # cases where two target argument (most frequent)
    sent_doc = insert_marker_of_two_arguments(tokens,deps,subj_token_start,subj_token_end,obj_token_start,
                                          obj_token_end,'@','$')
    return sent_doc, parse_legitimacy_check(sentence,words,sent_doc)

def get_deps_word_to_word(words):
    w2w_deps = []
    ts = [w.text for w in words]
    for w in words:
        if w.head > 0:
            w2w_deps.append((w.text,ts[w.head-1],w.deprel))
        else:
            w2w_deps.append((w.text,'root',w.deprel))
    return w2w_deps
    
def parse_legitimacy_check(original_sentence,parse_words,summary):
    #===== automatically check the correctness of built token-level data by verifying:
    # 1) if entity markers are placed at right positions in the token sequence;
    # 2) if dependencies are the same as the output of stanza using indexes from the token sequence with
    #    entity markers;
    # 3) if entity marker index are right
    #====================================
    ## words are stanza object: words
    w2w_deps_stanza = get_deps_word_to_word(parse_words)
    #1)
    tokens = summary["tokens"]

    sent = original_sentence.translate(replacements)
    sent = sent.replace('\xa0','')
    tmp = sent.replace(' ','') == ''.join(tokens).replace(' ','')
    if not tmp:
        print("entity markers were inserted at wrong positions.")
        return False
    
    #2)
    w2w_deps_summary = []
    for i1, i2, dep in summary["dependencies"]:
        if i2 > 0:
            w2w_deps_summary.append((tokens[i1-1],tokens[i2-1],dep))
        else:
            w2w_deps_summary.append((tokens[i1-1],"root",dep))
    tmp = w2w_deps_stanza == w2w_deps_summary
    if not tmp:
        print("error detected on dependencies using new token indexes.")
        #print(w2w_deps_stanza)
        #print("==========")
        #print(w2w_deps_summary)
        return False
    
    #3)
    if "subj" in summary and len(summary["subj"]) > 0:
        assert len(summary["subj"]) == 4, "number of entity markers must be 4."
        tmp = set([tokens[ix] for ix in summary["subj"]]) == {'@'}
        if not tmp:
            print("wrong subject indexes.")
            return False
    if "obj" in summary and len(summary["obj"]) > 0:
        assert len(summary["obj"]) == 4, "number of entity markers must be 4."
        tmp = set([tokens[ix] for ix in summary["obj"]]) == {'$'}
        if not tmp:
            print("wrong object indexes.")
            return False
    if "subj-obj" in summary and len(summary["subj-obj"]) > 0:
        assert len(summary["subj-obj"]) == 4, "number of entity markers must be 4."
        tmp = set([tokens[ix] for ix in summary["subj-obj"]]) == {'¢'}
        if not tmp:
            print("wrong subject-obj indexes.")
            return False
    return True

def transform(sent):
    # retag arguments of bert data files in previous formats into a unified one 
    if sent.count('<<') == 1 and sent.count('>>') == 1:
        return sent.replace('<<','$$').replace('>>','$$')
    elif sent.count('<<') == 1 and sent.count('>>') > 1:
        prev_c = ''
        found = False
        for i, c in enumerate(sent):
            if c == '<' and prev_c == '<':
                # index of first '<'
                start_ix = i - 1
                found = True
            if c == '>' and prev_c == '>' and found:
                # index of last '>'
                end_ix = i 
                break
            prev_c = c
        return sent[:start_ix] + '$$ ' + sent[start_ix+3:end_ix-2] + ' $$' + sent[end_ix+1:]
    else:
        raise ValueError("can't be 0")