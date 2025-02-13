import json
import logging
import os
import pickle
import random

from llm.data.tokenizer import Tokenizer


def process(input_path: str, output_path: str) -> None:
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    random.seed(2024)
    random.shuffle(data)
    
    size = len(data)
    
    pretrain_size = int(size * 0.5)
    finetune_size = int(size * 0.8)
    
    pretrain_data = data[:pretrain_size]
    finetune_data = data[pretrain_size:finetune_size]
    alignment_data = data[finetune_size:]
    
    
    pretrain_texts = []
    for item in pretrain_data:
        body = '\n'.join(item['paragraphs'])
        title = item['title']
        pretrain_texts.append(f'{title}\n{body}')
    
    pretrain_text = '\n\n'.join(pretrain_texts)
    logging.info(f'pretrain text: {pretrain_text[:100]}')
    
    finetune_texts = []
    instruction = '請用以下題目寫一首詩'
    instruction_label = '<INS>'
    input_label = '<INP>'
    response_label = '<RES>'
    
    for item in finetune_data:
        body = '\n'.join(item['paragraphs'])
        title = item['title']
        content = f'{instruction_label}{instruction}{input_label}{title}{response_label}{body}'
        finetune_texts.append(content)
    logging.info(f'The instruction finetune data is a list of formatted texts. Here is the first item: {finetune_texts[0]}')
    
    five_word_texts = []
    other_texts = []
    for item in alignment_data:
        if all(len(paragraph) == 12 for paragraph in item['paragraphs']):
            five_word_texts.append(item)
        else:
            other_texts.append(item)
    
    alignment_texts = []
    for postive_text in five_word_texts:
        positive_body =  '\n'.join(postive_text['paragraphs'])
        positive_title = postive_text['title']
        
        negative_text = random.choice(other_texts)
        negative_title = negative_text['title']
        negative_body = '\n'.join(negative_text['paragraphs'])
        
        positive_text = f'{instruction_label}{instruction}{input_label}{positive_title}{response_label}{positive_body}'
        negative_text = f'{instruction_label}{instruction}{input_label}{negative_title}{response_label}{negative_body}'
        alignment_texts.append((positive_text, negative_text))
    logging.info(f'The alignment data is a list of positive-negative pairs. Here is the first pair: {alignment_texts[0]}')
    
    corpus = [pretrain_text]
    corpus.extend(finetune_texts)
    for positive_text, negative_text in alignment_texts:
        corpus.extend([positive_text, negative_text])
    corpus.append('\0')
    corpus = ''.join(corpus)
    tokenizer = Tokenizer(corpus=corpus)
    
    pretrain_dataset = tokenizer.encode(pretrain_text)
    finetune_dataset = [[tokenizer.encode(text)] for text in finetune_texts]
    
    finetune_dataset = [torch.tensor(tokenizer.encode(text), dtype=torch.long) for text in finetune_texts]
    alignment_dataset = [(tokenizer.encode(positive_text), 
                          tokenizer.encode(negative_text))
                         for positive_text, negative_text in alignment_texts]

    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    with open(output_path, 'wb') as f:
        data = {
            'pretrain': pretrain_dataset,
            'finetune': finetune_dataset,
            'alignment': alignment_dataset,
        }
        pickle.dump(data, f)


if __name__ == '__main__':
    
    process('data.json', 'output')
    
    