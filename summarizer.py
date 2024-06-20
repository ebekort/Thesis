from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from huggingface_hub import login
import torch
from evaluate import load
import random
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re


def load_model(model_id):
    login('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')
    tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir="/scratch/s4143299/instruct/")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        cache_dir="/scratch/s4143299/instruct"
    )
    return tokenizer, model


def order_summaries(data):
    # sorting summaries
    ordered_sums = {key: sorted([(k.replace('rank', 'Sum'), k.replace('rank', 'exp'),v) for k, v in value.items() if 'rank' in k], key=lambda x:x[2]) for key, value in data.items()}

    return ordered_sums





def generate_zero(paper, dev_article=None, dev_sum1=None, dev_sum2=None, dev_sum3=None, dev_exp=None):
    messages = [
       {"role": "system", "content": """You are an expert in the field of natural language processing and you are summarizing scientific papers based on the novelty.
You always make a summary with the following structure: Introduction: a short introduction on the topic of the paper first,
Novelty: list novelty aspects using bullet points
conclusion: brief conclusion""" },
        {"role": "user", "content": f""" Your task is to generate a summary of a scientific paper on the novelty discussed in the paper. The novelty is considered to be the new insights, discoveries or approaches in comparison to the related work discussed in the paper. The summary has to be relevant for readers that are experts in Natural Language Processing who are interested in the novelty in the paper.

Paper:
{paper}

Summary:
"""
        }
    ]
    return messages


def generate_top(paper, dev_article, dev_sum1):
    messages = [
        {"role": "system", "content": """You are an expert in the field of natural language processing and you are summarizing scientific papers based on the novelty.
You always make a summary with the following structure: Introduction: a short introduction on the topic of the paper first,
Novelty: list novelty aspects using bullet points
conclusion: brief conclusion""" },
        {"role": "user", "content": f""" Your task is to generate a summary of a scientific paper on the novelty discussed in the paper. The novelty is considered to be the new insights, discoveries or approaches in comparison to the related work discussed in the paper. The summary has to be relevant for readers that are experts in Natural Language Processing who are interested in the novelty in the paper.

Paper:
{dev_article}
"""
        },
        {"role": "assistant", "content": dev_sum1},
        {"role": "user", "content": f"Paper:\n{paper}"},
    ]
    return messages


def generate_all(paper, dev_article, dev_sum1, dev_sum2, dev_sum3):
    messages = [
        {"role": "system", "content": """You are an expert in the field of natural language processing and you are summarizing scientific papers based on the novelty.
You always make a summary with the following structure: Introduction: a short introduction on the topic of the paper first,
Novelty: list novelty aspects using bullet points
conclusion: brief conclusion""" },
        {"role": "user", "content": f""" Your task is to generate three summaries of a scientific paper on the novelty discussed in the paper, the first should be the best paper and the last should be the worst. The novelty is considered to be the new insights, discoveries or approaches in comparison to the related work discussed in the paper. The summary has to be relevant for readers that are experts in Natural Language Processing who are interested in the novelty in the paper.
         
Paper:
{dev_article}
"""
        },
        {"role": "assistant", "content": f"Rank 1 summary:\n{dev_sum1}\n\nRank 2 summary:\n{dev_sum2}\n\nRank 3 summary:\n{dev_sum3}"},
        {"role": "user", "content": f"Paper:\n{paper}"},
    ]
    return messages


def generate_exp(paper, dev_article, dev_sum1, dev_sum2, dev_sum3, exp1, exp2, exp3):
    messages = [
        {"role": "system", "content": """You are an expert in the field of natural language processing and you are summarizing scientific papers based on the novelty.
You always make a summary with the following structure: Introduction: a short introduction on the topic of the paper first,
Novelty: list novelty aspects using bullet points
conclusion: brief conclusion""" },
        {"role": "user", "content": f""" Your task is to generate three summaries of a scientific paper on the novelty discussed in the paper, the first should rank the best and the last should rank the worst. Additionally, provide an explanation for the rank of each summary The novelty is considered to be the new insights, discoveries or approaches in comparison to the related work discussed in the paper. The summary has to be relevant for readers that are experts in Natural Language Processing who are interested in the novelty in the paper.

Paper:
{dev_article}
"""
        },
        {"role": "assistant", "content": f"Rank 1 summary:\n{dev_sum1}\n\nRank 2 summary:\n{dev_sum2}\n\nRank 3 summary:\n{dev_sum3}\n\nExplanation Rank 1 summary:\n{exp1}\n\nExplanation Rank 2 summary:\n{exp2}\n\nExplanation Rank 3 summary:\n{exp3}"},
        {"role": "user", "content": f"Paper:\n{paper}"},
    ]
    return messages



def generate_prompts(test, dev, ordered_dict):

    prompts = {
        'zero': generate_zero(paper=test['text']),
        'top': generate_top(paper=test['text'], dev_article=dev['text'], dev_sum1=dev[ordered_dict[0][0]]),
        'all': generate_all(paper=test['text'], dev_article=dev['text'], dev_sum1=dev[ordered_dict[0][0]], dev_sum2=dev[ordered_dict[1][0]], dev_sum3=dev[ordered_dict[2][0]]),
        'exp': generate_exp(paper=test['text'], dev_article=dev['text'], dev_sum1=dev[ordered_dict[0][0]], dev_sum2=dev[ordered_dict[1][0]], dev_sum3=dev[ordered_dict[2][0]],
                            exp1=dev[ordered_dict[0][1]], exp2=dev[ordered_dict[1][1]], exp3=dev[ordered_dict[2][1]])
    }

    return prompts


def remove_stop_words(text):
    #nltk.download('stopwords')

    words = word_tokenize(text)

    # Get the list of stopwords for English
    stop_words = set(stopwords.words('english'))

    # Remove stopwords from the text
    filtered_words = [word for word in words if word.lower() not in stop_words]

    # Join the filtered words back into a string
    filtered_text = ' '.join(filtered_words)

    return filtered_text


def clean_text(text): 
    # Remove special characters but keep normal punctuation
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s.,!?;\'\"-\(\)]', '', text)
    
    # Remove references in the format (name et al, date)
    
    cleaned_text = re.sub(r'\(([\w\s]+(et al.)?(and [\w\s]+)?\s*,\s*\d{4}[abcd,]*(;\s*)?)+\)|[\w\s]+(et al.)?(and [\w\s]+)?\s*\(\s*\d{4}[abcd,]*\)[s;]?|[\w\s]+et al\.(,|\s|$)|\b[A-Z][a-z]*\s+\(\d{4}\)', ' ', cleaned_text) # from https://regex101.com/r/kN6sD0/2
    # if 6 too long check that

    cleaned_text = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', ' ', cleaned_text) # remove urls https://regexr.com/38bj4

    # Remove extra spaces left by deletions
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    return cleaned_text


def filter_no_words(text):
    words = text.split()
    filtered_words = [word for word in words if re.search(r'\b[a-zA-Z]{2,}\b', word) and not re.search(r'[+=♂♀\^\*−]', word)] # filter lines that contain no words

    filtered_text = ' '.join(filtered_words)

    return filtered_text


def preprocess_data(data):
    for k,v in data.items():
        text = v['text']
        text = remove_stop_words(text)
        text = clean_text(text)
        text = filter_no_words(text)

        data[k]['text'] = text

    return data



def split_data(dev_ids=None, rand_seed=None):
    if dev_ids == None and rand_seed != None:
        random.seed(rand_seed)
        dev_ids = random.sample(range(25), 5)
    elif dev_ids is None and rand_seed is None:
        raise ValueError("Either dev_ids or rand_seed must be provided, not both None.")
    
    elif dev_ids is not None and rand_seed is not None:
        raise ValueError("Only one of dev_ids or rand_seed should be provided, not both.")

    test_ids = list(set(range(25)) - set(dev_ids))
    #test_ids = random.sample(remaining_numbers, 5) # delete this and return remaining numbers when doing full test

    return dev_ids, test_ids


def extract_summary(text):
    pattern = r'\*\*Best Summary\*\*\s*\n\n(.*?)\n\*\*Good Summary\*\*|\bRank 1 summary:\s*\n(.*?)\nRank 2 summary:'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        # Check which group was matched
        summary = match.group(1) if match.group(1) else match.group(2)
        summary = summary.strip()
    else:
        summary = text

    return summary


def evaluate(target, references, bertscore):
    
    pass





def main():
    bertscore = load("bertscore")
    with open('merged_data2.json', 'r') as inp:
        data = json.load(inp)

    data = preprocess_data(data)

    #dev_ids, test_ids = split_data(rand_seed=321) 
    dev_ids, test_ids = split_data(dev_ids=[14, 22, 9, 5, 17])
    print(f'Splitting data\ndev ids: {dev_ids}\ntest ids: {test_ids}')

    ordered_dict = order_summaries(data)
    refs = [data[str(i)][ordered_dict[str(i)][0][0]] for i in test_ids]

    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    tokenizer, model = load_model(model_id=model_id)

    output_dict = {}

    dev_ids=[5]
    
    for i in dev_ids:
        i = str(i)
        dev = data[i]
        output_dict[i] = {'zero': [], 'top': [], 'all': [], 'exp': []}

        for j in test_ids:
            j = str(j)
            test = data[j]

            prompt_dict = generate_prompts(test, dev, ordered_dict[i])

            for k,v in prompt_dict.items():

                input_ids = tokenizer.apply_chat_template(
                    v,
                    add_generation_prompt=True,
                    return_tensors="pt"
                ).to(model.device)
                prompt_length = input_ids.shape[-1]
                print(f'Generating summary using {k} summary/summaries\n\nPrompt length: {prompt_length}\n\n')

                if prompt_length > 8192:
                    print('context length too big')
                    output_dict[i][k].append('too big')
                    continue

                terminators = [
                    tokenizer.eos_token_id,
                    tokenizer.convert_tokens_to_ids("<|eot_id|>")
                ]

                outputs = model.generate(
                    input_ids,
                    eos_token_id=terminators,
                    do_sample=True,
                    max_new_tokens=3000,
                    temperature=0.6,
                    top_p=0.9,
                )
                response = outputs[0][input_ids.shape[-1]:]
                tokenized_sum = tokenizer.decode(response, skip_special_tokens=True)
                if k == 'all' or k == 'exp':
                    tokenized_sum = extract_summary(tokenized_sum)
                output_dict[i][k].append(tokenized_sum)
                print(f'Generated summary for {k} summary/summaries:\n\n{tokenized_sum}')

        #del tokenized_sum, response, outputs, input_ids
        #torch.cuda.empty_cache()

        
    for i in dev_ids:
        print(f'scores for prompts that used article {i} as dev article')
        for k in ['zero', 'top', 'all', 'exp']:
            score = bertscore.compute(predictions=output_dict[str(i)][k], references=refs, lang='en')
            #output_dict[k]['score'] = score['f1']
            print(f'Score for similarity between generated summary and top summary using {k}-shot learning: {score}')


    with open('out.json', 'w') as out:
        json.dump(output_dict, out)


if __name__ == "__main__":
    main()
