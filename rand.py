import bibtexparser
import random
import requests
import re
import json

with open('./bibtext.txt', 'r', encoding='utf-8') as inp:
    text = inp.read()

random.seed(123)

library = bibtexparser.parse_string(text)
nlp_articles = [entry for entry in library.entries if 'title' in entry and 'neural network' in entry['title'].lower() and
                'proceedings' not in entry['title'].lower() and 'proceeding' not in entry['title'].lower()]

random_articles = random.sample(nlp_articles, min(25, len(nlp_articles)))

random_articles.pop(9)
random_articles.pop(18)

selected_indices = [nlp_articles.index(article) for article in random_articles]

# Function to choose random articles not already in random_articles
def choose_random_articles(nlp_articles, selected_indices, num_articles):
    remaining_indices = [i for i in range(len(nlp_articles)) if i not in selected_indices]
    return random.sample(remaining_indices, min(num_articles, len(remaining_indices)))

# Choose 2 additional random articles
additional_indices = choose_random_articles(nlp_articles, selected_indices, 2)
additional_articles = [nlp_articles[index] for index in additional_indices]

# Add the additional articles to random_articles
random_articles.extend(additional_articles)

data = {}

for i, article in enumerate(random_articles):
  data[i] = {
      "title": re.sub(r'[{}]', '', article['title']),
      "url": article['url'],
  }

for dat in data:
  print(dat, data[dat])


'''

file_path = "data.json"




# Write dictionary to JSON file
with open(file_path, 'w') as json_file:
    json.dump(data, json_file)'''

