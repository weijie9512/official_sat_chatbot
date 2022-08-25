import pytorch_lightning as pl
import textdistance as td
import numpy as np
import argparse
import torch
from torch import nn
import torch.nn.functional as F
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel
)

from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    AutoModelWithLMHead,
    AutoTokenizer
)
from tokenizers import ByteLevelBPETokenizer

from tokenizers.processors import BertProcessing

from transformers import BertTokenizer, BertForSequenceClassification

# Initialisation
model = BertForSequenceClassification.from_pretrained('/home/wanhee/SATbot2.0/model/Final Model')
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

args_dict = dict(
    # Huggingface model
    model_name_or_path='/home/wanhee/SATbot2.0/model/Final Model',
    tokenizer_name_or_path='bert-base-chinese',
)

args = argparse.Namespace(**args_dict)


#load emotion classifier (T5 small)
with torch.no_grad():
    model.to(torch.device('cpu'))
    model.load_state_dict(torch.load('best_bert_base_chinese.pt', map_location=torch.device('cpu')), strict=False)

#load empathy classifier (T5 small)
with torch.no_grad():
    model.load_state_dict(torch.load('best_bert_base_chinese.pt', map_location=torch.device('cpu')), strict=False) #change path

#Load pre-trained GPT2 language model weights
with torch.no_grad():
    gptmodel = GPT2LMHeadModel.from_pretrained('gpt2')
    gptmodel.eval()

#Load pre-trained GPT2 tokenizer
gpttokenizer = GPT2Tokenizer.from_pretrained('gpt2')

#simple tokenizer + stemmer
regextokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
stemmer = nltk.stem.PorterStemmer()


def get_emotion(text):
    # ----- 3. Predict -----#
    # text = "今日心情好正"
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    X_test_tokenized = tokenizer(text, padding=True, truncation=True, max_length=512)

    b_input_ids= torch.tensor(X_test_tokenized['input_ids']).unsqueeze(0)
    b_attention_mask = torch.tensor(X_test_tokenized['attention_mask']).unsqueeze(0)

    with torch.no_grad():
        outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_attention_mask)

    logits = outputs[0]
    print(logits)
    logits = logits.detach().numpy()
    predict_label = np.argmax(logits, axis=1).flatten()
    print(predict_label)

    label = " "

    for i in predict_label:
        if i == 0:
            label = "唔開心"
        elif i == 1:
            label = "嬲"
        elif i == 2:
            label = "擔心"
        else:
            label = "開心"
        print(label)
    
    return label

def empathy_score(text):
  '''
  Computes a discrete numerical empathy score for an utterance (scale 0 to 2)
  '''
  with torch.no_grad():
      input_ids = emp_model.tokenizer.encode(text + '</s>', return_tensors='pt')
      output = emp_model.model.generate(input_ids=input_ids, max_length=2)
      dec = [emp_model.tokenizer.decode(ids) for ids in output]
  label = dec[0]
  if label == 'no':
    score = 0.0
  elif label == 'weak':
    score = 1.0
  else:
    score = 2.0
  #normalise between 0 and 1 dividing by the highest possible value:
  return score/2


def perplexity(sentence):
  '''
  Computes the PPL of an utterance using GPT2 LM
  '''
  tokenize_input = gpttokenizer.encode(sentence)
  tensor_input = torch.tensor([tokenize_input])
  with torch.no_grad():
      loss = gptmodel(tensor_input, labels=tensor_input)[0]
  return np.exp(loss.detach().numpy())


def repetition_penalty(sentence):
  '''
  Adds a penalty for each repeated (stemmed) token in
  an utterance. Returns the total penalty of the sentence
  '''
  word_list = regextokenizer.tokenize(sentence.lower())
  filtered_words = [word for word in word_list if word not in stopwords.words('english')]
  stem_list = [stemmer.stem(word) for word in filtered_words]
  penalty = 0
  visited = []
  for w in stem_list:
    if w not in visited:
      visited.append(w)
    else:
      penalty += 0.01
  return penalty


def fluency_score(sentence):
  '''
  Computes the fluency score of an utterance, given by the
  inverse of the perplexity minus a penalty for repeated tokens
  '''
  ppl = perplexity(sentence)
  penalty = repetition_penalty(sentence)
  score = (1 / ppl) - penalty
  #normalise by the highest possible fluency computed on the corpus:
  normalised_score = score / 0.16
  if normalised_score < 0:
    normalised_score = 0
  return round(normalised_score, 2)


def get_distance(s1, s2):
  '''
  Computes a distance score between utterances calculated as the overlap
  distance between unigrams, plus the overlap distance squared over bigrams,
  plus the overlap distance cubed over trigrams, etc (up to a number of ngrams
  equal to the length of the shortest utterance)
  '''
  s1 = re.sub(r'[^\w\s]', '', s1.lower()) #preprocess
  s2 = re.sub(r'[^\w\s]', '', s2.lower())
  s1_ws = regextokenizer.tokenize(s1) #tokenize to count tokens later
  s2_ws = regextokenizer.tokenize(s2)
  max_n = len(s1_ws) if len(s1_ws) < len(s2_ws) else len(s2_ws) #the max number of bigrams is the number of tokens in the shorter sentence
  ngram_scores = []
  for i in range(1, max_n+1):
    s1grams = nltk.ngrams(s1.split(), i)
    s2grams = nltk.ngrams(s2.split(), i)
    ngram_scores.append((td.overlap.normalized_distance(s1grams, s2grams))**i) #we normalize the distance score to be a value between 0 and 10, before raising to i
  normalised_dis = sum(ngram_scores)/(max_n) #normalised
  return normalised_dis


def compute_distances(sentence, dataframe):
  '''
  Computes a list of distances score between an utterance and all the utterances in a dataframe
  '''
  distances = []
  for index, row in dataframe.iterrows():
    df_s = dataframe['sentences'][index] #assuming the dataframe column is called 'sentences'
    distance = get_distance(df_s.lower(), sentence)
    distances.append(distance)
  return distances


def novelty_score(sentence, dataframe):
  '''
  Computes the mean of the distances beween an utterance
  and each of the utterances in a given dataframe
  '''
  if dataframe.empty:
    score = 1.0
  else:
    d_list = compute_distances(sentence, dataframe)
    d_score = sum(d_list)
    score = d_score / len(d_list)
  return round(score, 2)


def get_sentence_score(sentence, dataframe):
  '''
  Calculates how fit a sentence is based on its weighted empathy, fluency
  and novelty values
  '''
  empathy = empathy_score(sentence)
  fluency = fluency_score(sentence)
  novelty = novelty_score(sentence, dataframe)
  score = empathy + 0.75*fluency + 2*novelty
  return score
