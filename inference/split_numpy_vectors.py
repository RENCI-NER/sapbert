import argparse
import numpy as np
import pandas as pd
import sys
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--INPUT_METADATA_FILE', type=str, default='/babeldata/updated_mapping_other/name_ids.csv',
                        help='input babel terms and ids')
    parser.add_argument('--INPUT_VECTOR_FILE', type=str,
                        default='/babeldata/updated_mapping_other/babel_prediction_output.npz',
                        help='input babel compressed numpy array')
    parser.add_argument('--MATCH_TYPE', type=str, default='other',
                        help='the matching type key used to store input vector file in the compressed npz format')
    parser.add_argument('--CHUNK_SIZE', type=int, default=1000000,
                        help='chunk size to split input data')

    args = parser.parse_args()
    INPUT_METADATA_FILE = args.INPUT_METADATA_FILE
    INPUT_VECTOR_FILE = args.INPUT_VECTOR_FILE
    MATCH_TYPE = args.MATCH_TYPE
    CHUNK_SIZE = args.CHUNK_SIZE

    meta_base, meta_ext = os.path.splitext(INPUT_METADATA_FILE)
    vec_base, vec_ext = os.path.splitext(INPUT_VECTOR_FILE)
    vec_data = np.load(INPUT_VECTOR_FILE)
    vec_ary = vec_data[MATCH_TYPE]
    for i, csv_chunk in enumerate(pd.read_csv(INPUT_METADATA_FILE, dtype=str, chunksize=CHUNK_SIZE)):
        csv_chunk.to_csv(f'{meta_base}_{i}{meta_ext}', index=False)
        start_index = i * CHUNK_SIZE
        end_index = min((i + 1) * CHUNK_SIZE, len(vec_ary))
        data_chunk = vec_ary[start_index:end_index]
        np.savez_compressed(f'{vec_base}_{i}{vec_ext}', other=data_chunk)

    sys.exit(0)
