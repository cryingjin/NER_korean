# load glove
glove_model = Glove.load('glove_embedding.model')
print(f'Load glove_model...{str(glove_model)}')

# word dict
word_dict = {}
for word in glove_model.dictionary.keys():
    word_dict[word] = glove_model.word_vectors[glove_model.dictionary[word]]
print('Lengh of word dict... : ', len(word_dict))
