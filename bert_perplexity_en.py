import numpy as np
import torch
from transformers import BertTokenizer, BertForMaskedLM
DEVICE = 'cuda:0'

class Perplexity_Checker(object):
    def __init__(self, MODEL_PATH, MODEL_NAME, CACHE_DIR, device='cpu'):
        if MODEL_PATH:
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
            self.model = BertForMaskedLM.from_pretrained(MODEL_PATH)
        else:
            self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
            self.model = BertForMaskedLM.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
        self.model.to(device)
        self.model.eval()
        self.DEVICE = device

    def add_device(self, DEVICE):
        self.DEVICE = DEVICE
        self.model.to(DEVICE)

    def sentence_preprocese(self, text):
        tokenized_text = np.array(self.tokenizer.tokenize(text))
        find_sep = np.argwhere(tokenized_text == '[SEP]')
        segments_ids = np.zeros(tokenized_text.shape, dtype=int)
        if find_sep.size == 1:
            start_point = 1
        else:
            start_point = find_sep[0, 0] + 1
            segments_ids[start_point:] = 1

        end_point = tokenized_text.size - 1

        tokenized_text = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        masked_texts = []
        for masked_index in range(start_point, end_point):
            new_tokenized_text = np.array(tokenized_text, dtype=int)
            new_tokenized_text[masked_index] = self.tokenizer.convert_tokens_to_ids(['[MASK]'])[0]
            masked_texts.append(new_tokenized_text)

        segments_ids = np.tile(segments_ids, (end_point - start_point, 1))

        return masked_texts, segments_ids, start_point, end_point, tokenized_text[start_point:end_point]

    def perplexity(self, text):
        indexed_tokens, segments_ids, start_point, end_point, real_indexs = self.sentence_preprocese(text)
        #print(indexed_tokens)
        tokens_tensor = torch.LongTensor(indexed_tokens)
        segments_tensors = torch.LongTensor(segments_ids)

        tokens_tensor = tokens_tensor.to(self.DEVICE)
        segments_tensors = segments_tensors.to(self.DEVICE)

        with torch.no_grad():
            outputs = self.model(tokens_tensor, token_type_ids=segments_tensors)
            predictions = torch.softmax(outputs[0], -1)
            #print(predictions)

        total_perplexity = 0
        for i, step in enumerate(range(start_point, end_point)):
            total_perplexity += np.log(predictions[i, step, real_indexs[i]].item())

        total_perplexity = -total_perplexity / (end_point - start_point)
        return total_perplexity


if __name__ == '__main__':
    import gzh

    # 模型名字
    MODEL_NAME = 'bert-base-uncased'

    # 模型存放地址
    CACHE_DIR = './cache_en'

    # 或者直接写模型存放地址
    MODEL_PATH = ''
    text_formatter = lambda x: "[CLS]{} [SEP]".format(x)
    pchecker = Perplexity_Checker(MODEL_NAME=MODEL_NAME, MODEL_PATH=MODEL_PATH, CACHE_DIR=CACHE_DIR, device='cuda')

    text_ori = "I'll call you again later"
    text0 = "I'll phone you again later"
    text1 = "I'll describe you again later"
    text2 = "I'll visit you again later"
    text3 = "I'll shout you again later"
    text4 = "I'll shout to you again later"
    text5 = "I'll shouts to you again later"
    text6 = "I'll shouted to you again later"

    print("原句的困惑度为：",pchecker.perplexity(text_formatter(text_ori)))
    print("text0:", pchecker.perplexity(text_formatter(text0)))
    print("text1:", pchecker.perplexity(text_formatter(text1)))
    print("text2:", pchecker.perplexity(text_formatter(text2)))
    print("text3:", pchecker.perplexity(text_formatter(text3)))
    print("text4:", pchecker.perplexity(text_formatter(text4)))
    print("text5:", pchecker.perplexity(text_formatter(text5)))
    print("text6:", pchecker.perplexity(text_formatter(text6)))