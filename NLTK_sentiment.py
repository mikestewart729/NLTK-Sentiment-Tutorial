import nltk
from pprint import pprint

## Download specific resources for this exercise
#nltk.download([
#    "names",
#    "stopwords",
#    "state_union",
#    "twitter_samples",
#    "movie_reviews",
#    "averaged_perceptron_tagger",
#    "vader_lexicon",
#    "punkt"
#])

## Start by getting the words from a corpus
#words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]

## Create a stopwords filter
#stopwords = nltk.corpus.stopwords.words("english")

## Filter out the stopwords
#words = [w for w in words if w.lower() not in stopwords]

## Now try out NLTK's built in word tokenizer
#text = """
#For some quick analysis, creating a corpus could be overkill. 
#If all you need is a word list, 
#there are simpler ways to achieve that goal."""
#pprint(nltk.word_tokenize(text), width = 79, compact = True)

## Create a frequency distribution
#words = nltk.word_tokenize(text)
#fd = nltk.FreqDist(words)

#print(fd.most_common(3))
#fd.tabulate(3)

## Now look at concordances and collocations
#text = nltk.Text(nltk.corpus.state_union.words())
#text.concordance("america", lines = 5) # Dumps to the console output
## Make a usable kind of concordance list
#concordance_list = text.concordance_list("america", lines = 2)
#for entry in concordance_list:
#    print(entry.line)

## NLTK Text class is a shortcut to creating many useful things
#words = nltk.word_tokenize(
#    """Beautiful is better than ugly.
#    Explicit is better than implicit.
#    Simple is better than complex."""
#)

#text = nltk.Text(words)
#fd = text.vocab() # Equivalent to an nltk FreqDist
#fd.tabulate(3)

## Now explore collocations, that is, bigrams, trigrams, etc.
#words = [w for w in nltk.corpus.state_union.words() if w.isalpha()]
#finder = nltk.collocations.TrigramCollocationFinder.from_words(words)

#print(finder.ngram_fd.most_common(2)) # Note: This works, but tabulate throws a TypeError

## Now start using nltk pre-trained/built in sentiment analyzer
from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer() # Instantiate a sentiment analyzer
#print(sia.polarity_scores("Wow, NLTK is really powerful!"))

## Test the sentiment analyzer against real data
#tweets = [t.replace("://","//") for t in nltk.corpus.twitter_samples.strings()]

## Use polarity scores from sentiment analyzer to classify the tweets
from random import shuffle

def is_positive(tweet):
    return sia.polarity_scores(tweet)["compound"] > 0

## Explore some tweet sentiment scores
#shuffle(tweets)
#for tweet in tweets[:10]:
#    print(">", is_positive(tweet), tweet)

## Explore some movie review sentiments 
positive_review_ids = nltk.corpus.movie_reviews.fileids(categories=["pos"])
negative_review_ids = nltk.corpus.movie_reviews.fileids(categories=["neg"])
all_review_ids = positive_review_ids + negative_review_ids

## Redefine is_positive to work on a whole review of more than one sentence
from statistics import mean

def is_positive_review(review_id):
    text = nltk.corpus.movie_reviews.raw(review_id)
    scores = [
        sia.polarity_scores(sentence)["compound"]
        for sentence in nltk.sent_tokenize(text)
    ]
    return mean(scores) > 0

#shuffle(all_review_ids)
#correct = 0
#for review_id in all_review_ids:
#    if is_positive_review(review_id):
#        if review_id in positive_review_ids:
#            correct += 1
#    else:
#        if review_id in negative_review_ids:
#            correct += 1

#print(f"{correct / len(all_review_ids):.2%} correct")

## Note this does not perform very well. This is because this classifier is 
## optimized for social media data. In the following, we will customize
## the classifier and see if we cna get better performance.

## Filter out unwanted words 
unwanted = nltk.corpus.stopwords.words("english") # List stopwords
unwanted.extend([w.lower() for w in nltk.corpus.names.words()]) # List names

def skip_unwanted(pos_tuple):
    word, tag = pos_tuple
    if not word.isalpha() or word in unwanted:
        return False
    if tag.startswith("NN"):
        return False
    return True

positive_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["pos"]))
)]
negative_words = [word for word, tag in filter(
    skip_unwanted,
    nltk.pos_tag(nltk.corpus.movie_reviews.words(categories=["neg"]))
)]

## Create frequency distributions for positive and negative words
## First filter out common words from both distributions
positive_fd = nltk.FreqDist(positive_words)
negative_fd = nltk.FreqDist(negative_words)

common_set = set(positive_fd).intersection(negative_fd)

for word in common_set:
    del positive_fd[word]
    del negative_fd[word]

top_100_positive = {word for word, count in positive_fd.most_common(100)}
top_100_negative = {word for word, count in negative_fd.most_common(100)}

## Now engineer bigram features
positive_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(
    w for w in nltk.corpus.movie_reviews.words(categories=["pos"])
    if w.isalpha() and w not in unwanted
)
negative_bigram_finder = nltk.collocations.BigramCollocationFinder.from_words(
    w for w in nltk.corpus.movie_reviews.words(categories=["neg"])
    if w.isalpha() and w not in unwanted
)

## Train and use the classifier!
def extract_features(text):
    features = dict()
    wordcount = 0
    compound_scores = list()
    positive_scores = list()

    for sentence in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sentence):
            if word.lower() in top_100_positive:
                wordcount += 1
        compound_scores.append(sia.polarity_scores(sentence)["compound"])
        positive_scores.append(sia.polarity_scores(sentence)["pos"])

    # Add 1 to final compound score to always have something non-negative
    # Some classifiers do not work with negative numbers
    features["mean_compound"] = mean(compound_scores) + 1
    features["mean_positive"] = mean(positive_scores)
    features["wordcount"] = wordcount

    return features

features = [
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "pos")
    for review in nltk.corpus.movie_reviews.fileids(categories=["pos"])
]
features.extend([
    (extract_features(nltk.corpus.movie_reviews.raw(review)), "neg")
    for review in nltk.corpus.movie_reviews.fileids(categories=["neg"])
])

# Train the classifier
train_count = len(features) // 4
shuffle(features)
classifier = nltk.NaiveBayesClassifier.train(features[:train_count])
classifier.show_most_informative_features(10)
print(nltk.classify.accuracy(classifier, features[train_count:]))