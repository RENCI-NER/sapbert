import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
import json
import pandas as pd
from mesh import MESH
from scipy.spatial.distance import cdist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--MESH_FILE_PATH', type=str, default='/data/mesh_desc_2022.xml',
                        help='MESH terms and ids downloaded from '
                             'https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/')
    parser.add_argument('--MESH_PREDICT_FILE_PATH', type=str, default='/data/pubmed_mesh_prediction_output.npy',
                        help='SapBERT model inference output file for PUBMED MESH terms')
    parser.add_argument('--MODEL_FOLDER', type=str, default='/data/SapBERT-from-PubMedBERT-fulltext',
                        help='SapBERT model trained from PUBMEDBERT full text')
    parser.add_argument('--INPUT_FILE', type=str, default='/data/annotated_17687.txt',
                        help='Input file with span annotations in PubAnnotator format for inference')
    parser.add_argument('--PUBMED_OUTPUT_FILE', type=str, default='/data/annotated_17687_output.csv',
                        help='SapBERT model inference output file for INPUT_FILE')
    args = parser.parse_args()
    MESH_FILE_PATH = args.MESH_FILE_PATH
    MESH_PREDICT_FILE_PATH = args.MESH_PREDICT_FILE_PATH
    MODEL_FOLDER = args.MODEL_FOLDER
    INPUT_FILE = args.INPUT_FILE
    PUBMED_OUTPUT_FILE = args.PUBMED_OUTPUT_FILE

    mesh = MESH(MESH_FILE_PATH)
    mesh.load_mesh()
    all_names = [name.strip('\n') for name in mesh.names]
    all_ids = mesh.ids

    # load sapbert
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    model = AutoModel.from_pretrained(MODEL_FOLDER).cuda(0)

    all_reps_emb = np.load(MESH_PREDICT_FILE_PATH)
    # do inference with pubmed data
    pubmed_pairs = []
    with open(INPUT_FILE) as f:
        for line in f:
            data = json.loads(line)
            for item in data['denotations']:
                text = item['text']
                category = item['obj']
                pubmed_pairs.append((text, category))

    if not pubmed_pairs:
        print(f'data cannot be loaded from {INPUT_FILE}, exit', flush=True)
        exit(1)

    print(f'total data elements: {len(pubmed_pairs)}', flush=True)

    all_pubmed_text = [p[0] for p in pubmed_pairs]
    all_pubmed_ids = [p[1] for p in pubmed_pairs]

    print(len(all_pubmed_text), len(all_pubmed_ids))
    print (f'all_reps_emb shape: {all_reps_emb.shape}')
    df_data = []
    for i in tqdm(range(len(all_pubmed_text))):
        toks = tokenizer.batch_encode_plus([all_pubmed_text[i]],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.cuda(0)
        output = model(**toks_cuda)
        cls_rep = output[0][:, 0, :]
        dist = cdist(cls_rep.cpu().detach().numpy(), all_reps_emb)
        nn_index = np.argmin(dist)
        df_data.append([all_pubmed_text[i], all_pubmed_ids[i], all_names[nn_index], all_ids[nn_index]])

    df = pd.DataFrame(df_data, columns = ['Span text', 'Labelled Category',
                                          'Predicted MESH term', 'Predicted MESH ID'])
    df.to_csv(PUBMED_OUTPUT_FILE, index=False)
