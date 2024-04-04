import argparse
import numpy as np
import pandas as pd
import time
import sys
import os
import gc
import re
from utils import sapbert_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--INPUT_FILE_PATH', type=str, default='/babeldata/updated_mapping_gene/name_ids.csv',
                        help='babel terms and ids')
    parser.add_argument('--MODEL_FOLDER', type=str, default='/data/SapBERT-fine-tuned-babel',
                        help='SapBERT model trained from babel data')
    parser.add_argument('--CHUNK_SIZE', type=int, default=5200000,
                        help='chunk size to stream read csv input file. Set it to 0 to disable stream read')
    parser.add_argument('--BABEL_OUTPUT_FILE', type=str, default='/babeldata/updated_mapping_gene/'
                                                                 'babel_prediction_output',
                        help='SapBERT model inference output file for babel input file')
    parser.add_argument('--BABEL_OUTPUT_SHAPE_FILE', type=str, default='/babeldata/updated_mapping_gene/'
                                                                       'babel_prediction_output_shape.txt',
                        help='SapBERT model inference output shape file for babel input file')

    args = parser.parse_args()
    INPUT_FILE_PATH = args.INPUT_FILE_PATH
    MODEL_FOLDER = args.MODEL_FOLDER
    BABEL_OUTPUT_FILE = args.BABEL_OUTPUT_FILE
    BABEL_OUTPUT_SHAPE_FILE = args.BABEL_OUTPUT_SHAPE_FILE
    CHUNK_SIZE = args.CHUNK_SIZE

    # get the type key for the data to be saved in np.savez_compressed() from INPUT_FILE_PATH
    match = re.search(r'(?<=mapping_)\w+(?=/)', INPUT_FILE_PATH)
    if match:
        match_type = match.group(0)
    else:
        match_type = ''

    if CHUNK_SIZE > 0:
        idx = 0
        shape_base, shape_ext = os.path.splitext(BABEL_OUTPUT_SHAPE_FILE)
        for df in pd.read_csv(INPUT_FILE_PATH, dtype=str, usecols=['Name'], chunksize=CHUNK_SIZE):
            all_names = df.Name.tolist()
            start = time.time()
            _, _, all_reps_emb = sapbert_predict(MODEL_FOLDER, all_names)
            end = time.time()
            print(f"Time consumed in sapbert prediction for chunk {idx} is {end-start}", flush=True)

            with open(f'{shape_base}_{idx}{shape_ext}', 'w') as f:
                f.write(str(all_reps_emb.shape))

            if match_type:
                np.savez_compressed(f'{BABEL_OUTPUT_FILE}_{idx}.npz', **{match_type: all_reps_emb})
            else:
                np.save(f'{BABEL_OUTPUT_FILE}_{idx}.npy', all_reps_emb)
            count = gc.collect()
            if count > 0:
                print(f"{count} gc collected for chunk {idx}", flush=True)
            idx += 1
    else:
        df = pd.read_csv(INPUT_FILE_PATH, dtype=str, usecols=['Name'])
        all_names = df.Name.tolist()
        start = time.time()
        _, _, all_reps_emb = sapbert_predict(MODEL_FOLDER, all_names)
        end = time.time()
        print(f"Time consumed in sapbert prediction is {end - start}", flush=True)
        with open(BABEL_OUTPUT_SHAPE_FILE, 'w') as f:
            f.write(str(all_reps_emb.shape))

        if match_type:
            np.savez_compressed(f'{BABEL_OUTPUT_FILE}.npz', **{match_type: all_reps_emb})
        else:
            np.save(f'{BABEL_OUTPUT_FILE}.npy', all_reps_emb)

    sys.exit(0)
