import openai

openai.api_key = "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"

response = openai.Embedding.create(
    input="This is an example",
    engine="text-similarity-davinci-001")

""" import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_dir = os.path.join(parent_dir, 'ADReSSo')
example_file =  os.path.join(data_dir, 'ADReSSo21-diagnosis-train/diagnosis/train/audio/ad/transcription/adrso024')

from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')

text = []
with open(example_file, 'r') as f:
    text.append(f.read())
print(text)

#Our sentences we like to encode
#sentences = ['This framework generates embeddings for each input sentence',
'Sentences are passed as a list of string.',
'The quick brown fox jumps over the lazy dog.'

#Sentences are encoded by calling model.encode()

embeddings = model.encode(example_file)

#Print the embeddings
for sentence, embedding in zip(text, embeddings):
    print("Sentence:", sentence)
    print("Embedding:", embedding)
    print("")

 """