import time

import numpy as np
import torch


def extract_embeddings(model, data_loader, pool_mode, logger, device):
    embeddings = []
    true_labels = []
    steps = len(data_loader)
    with torch.no_grad():
        end = time.time()
        start = time.time()
        for batch_idx, batch in enumerate(data_loader):

            if torch.cuda.is_available():
                batch = {key: val.to(device) for key, val in batch.items()}
            labels = batch['labels']
            embeddings_dict = model.compute_embeddings(**batch)
            batch_embeddings = embeddings_dict[pool_mode].to('cpu').numpy()
            labels = labels.to('cpu').numpy()
            true_labels.append(labels)
            embeddings.append(batch_embeddings)
            batch_time = time.time() - end
            total_time = time.time() - start
            end = time.time()
            logger.info('{0}: [{1:>2d}/{2}] '
                        'Batch Time: {batch_time:.3f} '
                        'Total Time: {total_time:.3f} '.format(
                            'Extract embeddings',
                            batch_idx + 1,
                            steps + 1,
                            batch_time=batch_time,
                            total_time=total_time))
    embeddings = np.concatenate(embeddings, axis=0)
    true_labels = np.concatenate(true_labels, axis=0)
    return embeddings, true_labels


def extract_transformer_embedds(model,
                                data_loader,
                                pool_mode,
                                logger,
                                device='cuda'):
    true_labels = torch.Tensor()
    embeddings = torch.Tensor()
    steps = len(data_loader)
    with torch.no_grad():
        end = time.time()
        start = time.time()
        for batch_idx, batch in enumerate(data_loader):

            batch_labels = batch['labels']
            batch_lengths = batch['lengths']
            model_inputs = {key: val.to(device) for key, val in batch.items()}
            model_outputs = model(**model_inputs,
                                  output_hidden_states=True,
                                  return_dict=True)

            last_hidden_state = model_outputs.hidden_states[-1].detach().cpu()
            # batch_embeddings: batch_size * seq_length * embedding_dim

            batch_embedding_list = [emb for emb in last_hidden_state]
            # Remove class token and padding
            batch_length_list = [ll for ll in batch_lengths]

            filtered_embeddings = [
                emb[1:(length + 1), :]
                for emb, length in zip(batch_embedding_list, batch_length_list)
            ]

            if 'mean' in pool_mode:
                batch_embeddings = torch.stack(
                    [torch.mean(emb, dim=0) for emb in filtered_embeddings])

            # keep class token only
            if 'cls' in pool_mode:
                batch_embeddings = torch.stack(
                    [emb[0, :] for emb in batch_embedding_list])

            embeddings = torch.cat((embeddings, batch_embeddings), dim=0)
            true_labels = torch.cat((true_labels, batch_labels), dim=0)

            batch_time = time.time() - end
            total_time = time.time() - start
            end = time.time()
            logger.info('{0}: [{1:>2d}/{2}] '
                        'Batch Time: {batch_time:.3f} '
                        'Total Time: {total_time:.3f} '.format(
                            'Extract embeddings',
                            batch_idx + 1,
                            steps + 1,
                            batch_time=batch_time,
                            total_time=total_time))
    embeddings = embeddings.numpy()
    true_labels = true_labels.numpy()
    return embeddings, true_labels
