import networkx as nx
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
import json
import pandas as pd
from scipy.spatial.distance import cdist


# load MESH terms and ids - you need to download MESH terms from
# https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/
MESH_FILE_PATH = '../data/mesh_desc_2022.xml'
mesh = MESH(MESH_FILE_PATH)
mesh.load_mesh()
all_names = mesh.names
all_ids = mesh.ids

print(len(all_names), len(all_ids))
print(all_names[:10])
print(all_ids[:10])

# load sapbert
tokenizer = AutoTokenizer.from_pretrained("/projects/ner/software/sapbert/SapBERT-from-PubMedBERT-fulltext")
model = AutoModel.from_pretrained("/projects/ner/software/sapbert/SapBERT-from-PubMedBERT-fulltext") #.cuda(1)

bs = 128
all_reps = []
for i in tqdm(np.arange(0, len(all_names), bs)):
    toks = tokenizer.batch_encode_plus(all_names[i:i + bs],
                                       padding="max_length",
                                       max_length=25,
                                       truncation=True,
                                       return_tensors="pt")
    # toks_cuda = {}
    # for k,v in toks.items():
    #    toks_cuda[k] = v.cuda(1)
    # output = model(**toks_cuda)

    output = model(**toks)
    cls_rep = output[0][:, 0, :]

    all_reps.append(cls_rep.cpu().detach().numpy())
all_reps_emb = np.concatenate(all_reps, axis=0)
print (all_reps_emb.shape)

# do inference with pubmed data
PUBMED_FILE = '/projects/ner/datasets/pubmed_ds/split_11.txt'
pubmed_pairs = []
with open(PUBMED_FILE) as f:
    for line in f:
        data = json.loads(line)
        text = data['text']
        id = data['_id']
        mesh_id = data['mesh_id']
        pubmed_pairs.append((text, id, mesh_id))

if not pubmed_pairs:
    print(f'data cannot be loaded from {PUBMED_FILE}, exit', flush=True)
    exit(1)

print(f'total data elements: {len(pubmed_pairs)}', flush=True)
print(pubmed_pairs[:10])

all_pubmed_text = [p[0] for p in pubmed_pairs]
all_pubmed_ids = [p[1] for p in pubmed_pairs]
all_pubmed_mesh_ids = [p[2] for p in pubmed_pairs]

print(all_pubmed_text[:10])
print(all_pubmed_ids[:10])
print(all_pubmed_mesh_ids[:10])

df_data = []
for i in tqdm(np.arange(0, len(all_pubmed_text), bs)):
    toks = tokenizer.batch_encode_plus(all_pubmed_text[i:i+bs],
                                       padding="max_length", 
                                       max_length=25, 
                                       truncation=True,
                                       return_tensors="pt")
    output = model(**toks)
    cls_rep = output[0][:,0,:]
    dist = cdist(cls_rep.cpu().detach().numpy(), all_reps_emb)
    nn_index = np.argmin(dist)
    df_data.append([all_pubmed_ids[i], all_pubmed_mesh_ids[i], all_names[nn_index], all_ids[nn_index]])

df = pd.DataFrame(df_data, columns = ['Labelled ID', 'Labelled MESH ID',
                                      'Predicted MESH term', 'Predicted MESH ID'])
df.to_csv('../data/pubmed_split_11_prediction_output.csv')
