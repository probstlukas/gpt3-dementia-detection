#Installation guideline: https://spacy.io/usage/

import spacy

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_sm")

# Process whole documents
text = ("Ja, da gibt es so ein Dings, wie heißt das noch... Ach genau ein Computer. "
        "Google in 2007, few people outside of the company took him "
        "seriously. “I can tell you very senior CEOs of major American "
        "car companies would shake my hand and turn away because I wasn’t "
        "worth talking to,” said Thrun, in an interview with Recode earlier "
        "this week.")
doc = nlp(text)

# Analyze syntax
print("Noun phrases:", [chunk.text for chunk in doc.noun_chunks])
print("Adjectives:", [token.lemma_ for token in doc if token.pos_ == "ADJ"])
# Index sentences --> TODO: calc word amount of every sentence
for sent_i, sent in enumerate(doc.sents):
    for token in sent:
        print(sent_i, token.i, token.text)


# Find named entities, phrases and concepts
for entity in doc.ents:
    print(entity.text, entity.label_)
