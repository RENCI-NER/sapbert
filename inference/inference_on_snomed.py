import networkx as nx
from tqdm import tqdm
from Snomed import Snomed
from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch
from scipy.spatial.distance import cdist


SNOMED_PATH = '/projects/ner/datasets/snomed/SnomedCT_InternationalRF2_PRODUCTION_20190731T120000Z'
snomed = Snomed(SNOMED_PATH)
snomed.load_snomed()

snomed_sf_id_pairs = []

for snomed_id in tqdm(snomed.graph.nodes):
    node_descs = snomed.index_definition[snomed_id]
    for d in node_descs:
        snomed_sf_id_pairs.append((d, snomed_id))

print(len(snomed_sf_id_pairs))

print(snomed_sf_id_pairs[:10])

snomed_sf_id_pairs_100k = snomed_sf_id_pairs[:100] # for simplicity

all_names = [p[0] for p in snomed_sf_id_pairs_100k]
all_ids = [p[1] for p in snomed_sf_id_pairs_100k]

print(all_names[:10])

print(all_ids[:10])

tokenizer = AutoTokenizer.from_pretrained("/projects/ner/software/sapbert/SapBERT-from-PubMedBERT-fulltext")  
model = AutoModel.from_pretrained("/projects/ner/software/sapbert/SapBERT-from-PubMedBERT-fulltext") #.cuda(1)

bs = 128
all_reps = []
for i in tqdm(np.arange(0, len(all_names), bs)):
    toks = tokenizer.batch_encode_plus(all_names[i:i+bs], 
                                       padding="max_length", 
                                       max_length=25, 
                                       truncation=True,
                                       return_tensors="pt")
    #toks_cuda = {}
    #for k,v in toks.items():
    #    toks_cuda[k] = v.cuda(1)
    #output = model(**toks_cuda)
    
    output = model(**toks)
    cls_rep = output[0][:,0,:]
    
    all_reps.append(cls_rep.cpu().detach().numpy())
all_reps_emb = np.concatenate(all_reps, axis=0)

print (all_reps_emb.shape)
query = "cardiopathy"
query_toks = tokenizer.batch_encode_plus([query], 
                                       padding="max_length", 
                                       max_length=25, 
                                       truncation=True,
                                       return_tensors="pt")

query_output = model(**query_toks)
print(len(query_output[0]))
print(len(query_output[1]))

query_cls_rep = query_output[0][:,0,:]

print(query_cls_rep.shape)
dist = cdist(query_cls_rep.cpu().detach().numpy(), all_reps_emb)
nn_index = np.argmin(dist)
print ("predicted label:", snomed_sf_id_pairs_100k[nn_index])

