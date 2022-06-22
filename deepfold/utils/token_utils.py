def save_token_embeddings(vocab, embeds, save_path):
    """Save pretrained token vectors in a unified format, where the first line
    specifies the `number_of_tokens` and `embedding_dim` followed with all
    token vectors, one token per line."""
    with open(save_path, 'w') as writer:
        writer.write(f'{embeds.shape[0]} {embeds.shape[1]}\n')
        for idx, token in enumerate(vocab.idx_to_token):
            vec = ' '.join(['{:.4f}'.format(x) for x in embeds[idx]])
            writer.write(f'{token} {vec}\n')
    print(f'Pretrained embeddings saved to: {save_path}')
