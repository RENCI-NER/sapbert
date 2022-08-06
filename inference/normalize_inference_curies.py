import pandas as pd
import json
import requests
import argparse
import warnings

warnings.filterwarnings('ignore', '.*ssl*', )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file', type=str,
                        default='../data/pubmed_split_11_prediction_output.csv',
                        help='input model prediction file that contains predicted MESH IDs')
    parser.add_argument('--MESH_ID_COLUMN_NAME', type=str,
                        default='Predicted MESH ID',
                        help='MESH ID column name in the input file to read in')
    parser.add_argument('--output_file', type=str,
                        default='../data/normalized_split_11_predictions.json',
                        help='output file that contains output from node normalizer to each input MESH ID')


    args = parser.parse_args()
    input_file = args.input_file
    mesh_id_col = args.MESH_ID_COLUMN_NAME
    output_file = args.output_file

    input_df = pd.read_csv(input_file, usecols=[mesh_id_col])
    print(input_df)
    input_df[mesh_id_col] = 'MESH:' + input_df[mesh_id_col]
    input_list = list(input_df[mesh_id_col])
    node_normalizer_url = 'https://nodenormalization-sri.renci.org/1.2/get_normalized_nodes'
    post_data = {
        "curies": input_list,
        "conflate": True
    }
    result = requests.post(node_normalizer_url, json=post_data)
    print(result.status_code)
    # json_object = json.dumps(dictionary, indent=4)
    with open(output_file, "w") as outf:
        json.dump(result.json(), outf, indent=2)
