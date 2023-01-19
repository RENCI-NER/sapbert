import argparse
from tqdm import tqdm
import numpy as np
import json
import pandas as pd
from mesh import MESH
from scipy.spatial.distance import cdist
from .utils import sapbert_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--MESH_FILE_PATH', type=str, default='/data/mesh_desc_2022.xml',
                        help='MESH terms and ids downloaded from '
                             'https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/')
    parser.add_argument('--MODEL_FOLDER', type=str, default='/data/SapBERT-from-PubMedBERT-fulltext',
                        help='SapBERT model trained from PUBMEDBERT full text')
    parser.add_argument('--PUBMED_FILE', type=str, default='/data/test_pubmed_split_11.txt',
                        help='PUBMED DS test file for inference')
    parser.add_argument('--PUBMED_OUTPUT_FILE', type=str, default='/data/pubmed_split_11_prediction_output.csv',
                        help='SapBERT model inference output file for PUBMED_FILE input file')
    args = parser.parse_args()
    MESH_FILE_PATH = args.MESH_FILE_PATH
    MODEL_FOLDER = args.MODEL_FOLDER
    PUBMED_FILE = args.PUBMED_FILE
    PUBMED_OUTPUT_FILE = args.PUBMED_OUTPUT_FILE

    mesh = MESH(MESH_FILE_PATH)
    mesh.load_mesh()
    all_names = [name.strip('\n') for name in mesh.names]
    all_ids = mesh.ids

    model, tokenizer, all_reps_emb = sapbert_predict(MODEL_FOLDER, all_names, use_gpu=False)

    # do inference with pubmed data
    pubmed_pairs = []
    with open(PUBMED_FILE) as f:
        for line in f:
            data = json.loads(line)
            text = data['text']
            id = data['_id']
            mesh_ids = []
            for mention in data['mentions']:
                mesh_ids.append(mention['mesh_id'])
            mesh_id = '|'.join(mesh_ids)
            pubmed_pairs.append((text, id, mesh_id))

    if not pubmed_pairs:
        print(f'data cannot be loaded from {PUBMED_FILE}, exit', flush=True)
        exit(1)

    print(f'total data elements: {len(pubmed_pairs)}', flush=True)

    all_pubmed_text = [p[0] for p in pubmed_pairs]
    all_pubmed_ids = [p[1] for p in pubmed_pairs]
    all_pubmed_mesh_ids = [p[2] for p in pubmed_pairs]

    print(len(all_pubmed_text), len(all_pubmed_ids), len(all_pubmed_mesh_ids))
    print(len(all_names), len(all_ids))
    print (f'all_reps_emb shape: {all_reps_emb.shape}')
    df_data = []
    for i in tqdm(range(len(all_pubmed_text))):
        toks = tokenizer.batch_encode_plus([all_pubmed_text[i]],
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
    df.to_csv(PUBMED_OUTPUT_FILE, index=False)
