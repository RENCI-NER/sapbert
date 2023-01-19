import argparse
import numpy as np
import pandas as pd
from utils import sapbert_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--INPUT_FILE_PATH', type=str, default='/data/babel/mapping/name_ids.csv',
                        help='babel terms and ids')
    parser.add_argument('--MODEL_FOLDER', type=str, default='/data/SapBERT-fine-tuned-babel',
                        help='SapBERT model trained from babel data')
    parser.add_argument('--BABEL_OUTPUT_FILE', type=str, default='/data/babel/babel_prediction_output.npy',
                        help='SapBERT model inference output file for babel input file')
    parser.add_argument('--BABEL_OUTPUT_SHAPE_FILE', type=str, default='/data/babel_prediction_output_shape.txt',
                        help='SapBERT model inference output shape file for babel input file')

    args = parser.parse_args()
    INPUT_FILE_PATH = args.INPUT_FILE_PATH
    MODEL_FOLDER = args.MODEL_FOLDER
    BABEL_OUTPUT_FILE = args.BABEL_OUTPUT_FILE
    BABEL_OUTPUT_SHAPE_FILE = args.BABEL_OUTPUT_SHAPE_FILE

    df = pd.read_csv(INPUT_FILE_PATH, dtype=str, usecols=['Name'])

    all_names = df.Name.tolist()

    _, _, all_reps_emb = sapbert_predict(MODEL_FOLDER, all_names)

    with open(BABEL_OUTPUT_SHAPE_FILE, 'w') as f:
        f.write(str(all_reps_emb.shape))

    np.save(BABEL_OUTPUT_FILE, all_reps_emb)
