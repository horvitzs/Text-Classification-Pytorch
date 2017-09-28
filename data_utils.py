import torch 
import pytorch_text.torchtext.vocab as vocab
import os 
import sys


glove = vocab.GloVe(name='6B', dim=100)

class Dictionary(object):
    def __init__(self):
        self.word2idx = glove.stoi
        self.idx2word = glove.itos 
        self.vectors = glove.vectors

    def __len__(self):
        return len(self.word2idx)


class Corpus_20News(object):
    def __init__(self):
        self.dictionary = Dictionary()


    def get_data(self, filename, max_length, batch_size=20):

        words = filename.split()
        tokens = len(words)

        # Tokenize the file content 
        ids = torch.LongTensor(tokens)
        for idx, word in enumerate(words):
            try: 
                ids[idx] = self.dictionary.word2idx[word]
            except KeyError:
                continue 
            if idx > max_length:
                break 
        num_batches = max_length // batch_size
        ids = ids[:num_batches * batch_size]
       
        return ids.view(batch_size, -1)

    
class DataLoader_20News(object):

    def load_data_labels(self, TEXT_DATA_DIR):
        texts = []  # list of text samples
        labels_index = {}  # dictionary mapping label name to numeric id
        labels = []  # list of label ids
        for name in sorted(os.listdir(TEXT_DATA_DIR)):
            path = os.path.join(TEXT_DATA_DIR, name)
            if os.path.isdir(path):
                label_id = len(labels_index)
                labels_index[name] = label_id
                for fname in sorted(os.listdir(path)):
                    if fname.isdigit():
                        fpath = os.path.join(path, fname)
                        if sys.version_info < (3,):
                            f = open(fpath)
                        else:
                            f = open(fpath, encoding='latin-1')
                        t = f.read()
                        i = t.find('\n\n')  # skip header
                        if 0 < i:
                            t = t[i:]
                        texts.append(t)
                        f.close()
                        labels.append(label_id)
        return texts, labels, labels_index

