import spacy

nlp = spacy.load("en_core_web_lg")

w1 = "red"
w2 = "yellow"

w1 = nlp.vocab[w1]
w2 = nlp.vocab[w2]

print(w1.similarity(w2))

s1 = nlp("I believe in the god of the Bible")
s2 = nlp("I trust in Christianity")
s3 = nlp("This weekend John will drink a beer")

print(s1.similarity(s2))
print(s1.similarity(s3))

# for token in s1:
#     print(token.text, token.pos_)

s1_verbs = " ".join([token.lemma_ for token in s1 if token.pos_ == "VERB"])
s1_nouns = " ".join([token.lemma_ for token in s1 if token.pos_ == "NOUN" or token.pos_ == "PROPN"])
s1_adjectives = " ".join([token.lemma_ for token in s1 if token.pos_ == "ADJ"])

s2_verbs = " ".join([token.lemma_ for token in s2 if token.pos_ == "VERB"])
s2_nouns = " ".join([token.lemma_ for token in s2 if token.pos_ == "NOUN" or token.pos_ == "PROPN"])
s2_adjectives = " ".join([token.lemma_ for token in s2 if token.pos_ == "ADJ"])

s3_verbs = " ".join([token.lemma_ for token in s3 if token.pos_ == "VERB"])
s3_nouns = " ".join([token.lemma_ for token in s3 if token.pos_ == "NOUN" or token.pos_ == "PROPN"])
s3_adjectives = " ".join([token.lemma_ for token in s3 if token.pos_ == "ADJ"])

# print(s1_verbs)
print(s2_nouns)
# print(s1_adjectives)

print(f"{s1} and {s2} NOUNS: {nlp(s1_nouns).similarity(nlp(s2_nouns))}")
print(f"{s1} and {s3} VERBS: {nlp(s1_verbs).similarity(nlp(s3_verbs))}")
print(f"{s2} and {s3} VERBS: {nlp(s2_verbs).similarity(nlp(s3_verbs))}")
