import nltk
from nltk.corpus import wordnet as wn

# Download the WordNet data
nltk.download('wordnet')

# Get the synset for dog
dog_synset = wn.synset('dog.n.01')

# Get all hyponyms of dog synset (all breeds of dogs as per WordNet hierarchy)
dog_breeds_synsets = list(dog_synset.closure(lambda s: s.hyponyms()))

# Extract WordNet IDs (wnids) from synsets
dog_wnids = [breed.name() for breed in dog_breeds_synsets]

print(dog_wnids)
print(len(dog_wnids))


# Extract breed names from synsets
dog_breeds = [breed.name().split('.')[0] for breed in
              dog_breeds_synsets]

print(sorted(dog_breeds))
print(len(dog_breeds))

