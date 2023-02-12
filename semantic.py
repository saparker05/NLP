

import spacy

nlp = spacy.load('en_core_web_md')

word1 = nlp('cat')
word2 = nlp('monkey')
word3 = nlp('banana')

print(word1.similarity(word2))
print(word3.similarity(word1))
print(word3.similarity(word2))

'''
Running the code above returns:
0.5929929675536907
0.22358825939615987
0.4041501317354622

Cat and monkey are the most similar, presumably as they are both animals. The similarity value returned for monkey 
and banana is almost double that of cat and banana. This makes sense as monkeys are known to eat bananas so there is 
a stronger link than between cats and bananas. 

I have added an example of my own below using the words fish, water and rabbit. It returns the following:

0.5115877574421515
0.3946717257021854
0.14215386519997833

Fish and water are the most similar as fish live in water. The value for fish and rabbit is a little bit less, they
are both living things so there is a similarity there too. Finally rabbit and water had a low value of similarity. 

'''
print("\n")

word1 = nlp('fish')
word2 = nlp('water')
word3 = nlp('rabbit')

print(word1.similarity(word2))
print(word3.similarity(word1))
print(word3.similarity(word2))

print("\n")

tokens = nlp('cat apple monkey banana ')

for token1 in tokens:
    
    for token2 in tokens:

        print(token1.text, token2.text, token1.similarity(token2))

print("\n")


sentence_to_compare = "Why is my cat on the car"

sentences = ["Where did my dog go",
             "Hello, there is my car",
             "I\'ve lost my car in my car",
             "I\'d like my boat back",
             "I will name my dog Diana"]

model_sentence = nlp(sentence_to_compare)

for sentence in sentences:
    similarity = nlp(sentence).similarity(model_sentence)
    print(sentence + " - ", similarity)


'''
I ran the file example.py twice, the first time without making any changes and the second time I changed the language model to 'en_core_web_sm'. 
The seond time a user warning appeared: [W007] The model you're using has no word vectors loaded, so the result of the Doc.similarity method 
will be based on the tagger, parser and NER, which may not give useful similarity judgements. This may happen if you're using one of the small 
models, e.g. `en_core_web_sm`, which don't ship with word vectors and only use context-sensitive tensors. You can always add your own word vectors, 
or use one of the larger models instead if available.

I noticed that the similarity values returned were much lower when using the second model. The first time the similarities returned when comparing
recipes with recipes and complaints with complaints were all high, most were around 0.8/0.9 with the lowest around 0.75. Comparing 
recipes to complaints returned similarity values between 0.4 and 0.85. 

In comparison, the second model returned similarity values of recipes against recipes and complaints against complaints of around 0.6/0.7 for most
of the comparions. The values when comparing recipes against complaints were between 0.1 and 0.8.
'''
