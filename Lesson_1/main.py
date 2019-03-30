from textblob import TextBlob
from textblob import Word
text = TextBlob("Tokenization is the first step in text analytics. The process of breaking down a text \
paragraph into smaller chunks such as words or sentence is called Tokenization. Token is a single \
entity that is building blocks for sentence or paragraph.")
#print(text.sentences)
sent1 = text.sentences[0]
words = sent1.words
#print(words)

# it is possible to determine lemma, canonical form, of a given word
w = Word('running')
lemma_w = w.lemmatize('v') # 'v' stands for verb
#print(lemma_w)


# let us extract noun phrases from the given text
noun_phrases = text.noun_phrases
#print(noun_phrases)

# part-of-speech tagging
pos_tags = text.tags
#print(pos_tags)

# n-grams
ngram_2 = text.ngrams(2)
#print(ngram_2)

# TextBlob has a sentiment analysis feature
sentiment = text.sentiment
#print(sentiment)

# TextBlob has a spell check feature, which provides possible corrections of the given spelling
word = Word('appel')
#print(word.spellcheck())

# spelling correction of a whole sentence
misspelled = TextBlob("Yo boy, gimme dat apple and lemme go")
corrected = misspelled.correct()
#print(corrected)

# text summarization
import random
nouns_phrases = text.noun_phrases

#print('This text is about...')
#for item in random.sample(noun_phrases, 5):
#    print(item)

# Translation and language detection
word = TextBlob('ғажайып')
#print(text.detect_language())

# let us translate the above phrase in Kazakh into English
#print(word.translate(to='en'))

# text classication using TextBlog. First we need to provide a training and a testing data
training = [
('He played football yesterday', 'neg'),
('He runs everyday', 'neg'),
('Alice saw bear the other day', 'neg'),
('Jimmy always feels happy', 'neg'),
('Carter will join the chess club', 'pos'),
('Emily will not let anybody to inslut her', 'pos'),
]
testing = [
('Bratt explained the situation to Cooper', 'neg'),
('Maria will give witness tomorrow', 'pos'),
('Christina enjoys singing', 'neg'),
('Matt will not attent convocation', 'pos')
]
# the reasoning behind these datasets has to do with tense of a sentence. Past and present tense sentences are
# labeled as 'neg', while future tense sentences are labeled as 'pos'

# import classifiers
from textblob import classifiers
# train Naive Bayes classifier
nb_classifier = classifiers.NaiveBayesClassifier(training)
# train decision tree classifier
dt_classifier = classifiers.DecisionTreeClassifier(training)

# test Naive Bayes classifier
nb_accuracy = nb_classifier.accuracy(testing)
#print(nb_accuracy)
# show informative features of Naive Bayes classifier
#nb_classifier.show_informative_features()

# test decision tree classifier
dt_accuracy = dt_classifier.accuracy(testing)
print(dt_accuracy)

# let us classify a new sentence using dt_classifier
sentence = TextBlob('Amanda rocked the concert', classifier=dt_classifier)
print(sentence.classify())
