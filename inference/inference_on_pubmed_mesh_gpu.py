import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import numpy as np
from mesh import MESH


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--MESH_FILE_PATH', type=str, default='/data/mesh_desc_2022.xml',
                        help='MESH terms and ids downloaded from '
                             'https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/')
    parser.add_argument('--MODEL_FOLDER', type=str, default='/data/SapBERT-from-PubMedBERT-fulltext',
                        help='SapBERT model trained from PUBMEDBERT full text')
    parser.add_argument('--PUBMED_OUTPUT_FILE', type=str, default='/data/pubmed_mesh_prediction_output.npy',
                        help='SapBERT model inference output file for MESH_FILE_PATH input file')
    parser.add_argument('--PUBMED_OUTPUT_SHAPE_FILE', type=str, default='/data/pubmed_mesh_prediction_output_shape.txt',
                        help='SapBERT model inference output shape file for MESH_FILE_PATH input file')
    args = parser.parse_args()
    MESH_FILE_PATH = args.MESH_FILE_PATH
    MODEL_FOLDER = args.MODEL_FOLDER
    PUBMED_OUTPUT_FILE = args.PUBMED_OUTPUT_FILE
    PUBMED_OUTPUT_SHAPE_FILE = args.PUBMED_OUTPUT_SHAPE_FILE

    mesh = MESH(MESH_FILE_PATH)
    mesh.load_mesh()

    all_names = [name.strip('\n') for name in mesh.names]
    all_ids = mesh.ids

    # load sapbert
    tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    model = AutoModel.from_pretrained(MODEL_FOLDER).cuda(0)

    bs = 128
    all_reps = []
    for i in tqdm(np.arange(0, len(all_names), bs)):
        toks = tokenizer.batch_encode_plus(all_names[i:i + bs],
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k,v in toks.items():
           toks_cuda[k] = v.cuda(0)
        output = model(**toks_cuda)

        cls_rep = output[0][:, 0, :]

        all_reps.append(cls_rep.cpu().detach().numpy())
    all_reps_emb = np.concatenate(all_reps, axis=0)

    with open(PUBMED_OUTPUT_SHAPE_FILE, 'w') as f:
        f.write(str(all_reps_emb.shape))

    np.save(PUBMED_OUTPUT_FILE, all_reps_emb)
