import bibtexparser
import random
import re
import json


def choose_random_articles(nlp_articles, selected_indices, num_articles):
  """
    Choose random articles from a list, excluding already selected ones.
    
    Args:
        nlp_articles (list): List of articles.
        selected_indices (list): Indices of already selected articles.
        num_articles (int): Number of articles to choose.
    
    Returns:
        list: Indices of randomly chosen articles.
  """
  remaining_indices = [i for i in range(len(nlp_articles)) if i not in selected_indices]
  return random.sample(remaining_indices, min(num_articles, len(remaining_indices)))


def filter_articles(bibtext):
  """
  Filter articles based on the presence of 'neural network' in the title.
    
  Args:
    bibtext (str): String containing bibtex entries.
    
  Returns:
    list: Filtered list of articles.
  """
  library = bibtexparser.parse_string(bibtext)
  filtered_articles = [entry for entry in library.entries if 'title' in entry and 'neural network' in entry['title'].lower() and
                  'proceedings' not in entry['title'].lower() and 'proceeding' not in entry['title'].lower()]
  
  return filtered_articles


def rand_articles(filtered_articles, seed=123):
  """
  Select a random sample of articles, with specific exclusions and replacements.
    
  Args:
    filtered_articles (list): List of filtered articles.
    seed (int): Random seed for reproducibility.
    
  Returns:
    list: List of selected random articles.
  """
  random.seed(seed)
  random_articles = random.sample(filtered_articles, min(25, len(filtered_articles)))

  # remove non engelish articles, done by manual inspecttion
  random_articles.pop(9)
  random_articles.pop(18)

  selected_indices = [filtered_articles.index(article) for article in random_articles]



  # Choose 2 additional random articles
  additional_indices = choose_random_articles(filtered_articles, selected_indices, 2)
  additional_articles = [filtered_articles[index] for index in additional_indices]

  # Add the additional articles to random_articles
  random_articles.extend(additional_articles)

  print(len(random_articles))

  # remove articles with wrong structure, done by manual inspection
  wrong_structure_indices = [0, 14, 16, 18, 20, 22]
  for i in range(len(wrong_structure_indices)):
    random_articles.pop(wrong_structure_indices[i]-i)

  additional_indices = choose_random_articles(filtered_articles, selected_indices, 6)
  additional_articles = [filtered_articles[index] for index in additional_indices]

  # Add the additional articles to random_articles
  random_articles.extend(additional_articles)

  return random_articles

def main():
  try:
    with open('./bibtext.txt', 'r', encoding='utf-8') as inp:
      text = inp.read()
  except FileNotFoundError:
    print("Error: The file 'bibtext.txt' was not found.")
    return

  filtered_articles = filter_articles(text)

  random_articles = rand_articles(filtered_articles)

  # adding articles in a dictionary and removing special characters from titles
  data = {}
  for i, article in enumerate(random_articles):
    data[i] = {
        "title": re.sub(r'[{}]', '', article['title']),
        "url": article['url'],
    }

  print('selected the following random articles:')
  for dat in data:
    print(dat, data[dat])

  with open('test.json', 'w') as json_file:
      json.dump(data, json_file)
