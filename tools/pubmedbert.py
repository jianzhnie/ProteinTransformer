import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

if __name__ == '__main__':
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract'
    model_name = 'microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        'cuda')

    text = f'Heart disease is {tokenizer.mask_token} leading cause of death in the United States.'
    tokenized = tokenizer(text, return_tensors='pt').to('cuda')
    print(tokenizer.convert_ids_to_tokens(tokenized.input_ids.squeeze()))
    output = model(**tokenized, return_dict=True)
    output.logits.size()
    print(
        tokenizer.convert_ids_to_tokens(
            torch.topk(output.logits[0, 4, :], 10).indices))

    model_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(
        'cuda')

    tokenized = tokenizer(text, return_tensors='pt').to('cuda')
    print(tokenizer.convert_ids_to_tokens(tokenized.input_ids.squeeze()))
    output = model(**tokenized, return_dict=True)
    print(
        tokenizer.convert_ids_to_tokens(
            torch.topk(output.logits[0, 4, :], 10).indices))
