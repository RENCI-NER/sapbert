import pandas as pd
import requests
import argparse
import warnings

warnings.filterwarnings('ignore', '.*ssl*', )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file', type=str,
                        default='../data/annotated_17687_output.csv',
                        help='input model prediction file that contains predicted MESH IDs')
    parser.add_argument('--MESH_ID_COLUMN_NAME', type=str,
                        default='Predicted MESH ID',
                        help='MESH ID column name in the input file to read in')
    parser.add_argument('--REQUEST_LIST_SIZE', type=int,
                        default=100,
                        help='MESH ID column name in the input file to read in')
    parser.add_argument('--output_file', type=str,
                        default='../data/annotated_17687_output_biolink.csv',
                        help='output file that contains output from node normalizer to each input MESH ID')


    args = parser.parse_args()
    input_file = args.input_file
    mesh_id_col = args.MESH_ID_COLUMN_NAME
    request_list_size = args.REQUEST_LIST_SIZE
    output_file = args.output_file

    input_df = pd.read_csv(input_file)
    print(input_df.shape)
    input_df[mesh_id_col] = 'MESH:' + input_df[mesh_id_col]
    input_set = list(set(list(input_df[mesh_id_col])))
    set_size = len(input_set)
    print(set_size)
    node_normalizer_url = 'https://nodenormalization-sri.renci.org/1.3/get_normalized_nodes'
    result_dict = {}
    count = 0
    while count < set_size:
        low_bound = count
        high_bound = low_bound + request_list_size
        try:
            result = requests.post(node_normalizer_url, json={
                "curies": input_set[low_bound:high_bound],
                "conflate": True
            })
            result_dict.update(result.json())
        except Exception as ex:
            print(ex)
        count += request_list_size
    # with open(f"../data/result_dict.json", "w") as outfile:
    #    json.dump(result_dict, outfile)
    input_df['Predicted Category'] = input_df[mesh_id_col].map(
        lambda x: result_dict[x]['type'][0] if result_dict[x] and 'type' in result_dict[x] else '')
    input_df.to_csv(output_file, index=False)
