import pickle
from pathlib import Path
import numpy as np


project_dir = Path(__file__).resolve().parent.parent
datasets_dir = project_dir.joinpath('./data/')
WORD2ID_dir = datasets_dir.joinpath('word2id.pkl')


def insert_embed(vocab_size):
    pre_embed = np.zeros((vocab_size,300))
    pre_embed[1,:]=np.random.randn(1,300)
    pre_embed[2,:]=np.random.randn(1,300)
    pre_embed[3,:]=np.random.randn(300)
    with open(WORD2ID_dir,'rb') as r:
        word2id = pickle.load(r)
        with open(datasets_dir.joinpath('sgns.weibo.word'),'r',encoding='utf-8') as e:
            word_vec = e.readlines()
            vec_dict = {}
            for k in word_vec:
                    w,v = k.strip().split(' ',1)
                    vec_dict[w] = v
            for k in word2id:
                    if k in vec_dict:
                        pre_embed[word2id[k],:]=np.fromstring(vec_dict[k],dtype='float',sep=' ')
                    else:
                        pre_embed[word2id[k],:]=np.random.randn(300)
    return(pre_embed)
            