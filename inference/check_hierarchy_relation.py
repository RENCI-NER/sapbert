import pandas as pd
import requests
import argparse
import warnings
import os


warnings.filterwarnings('ignore', '.*ssl*', )


def get_hierarchy_relation(input_type):
    lookup_url = 'https://bl-lookup-sri.renci.org/bl/{input_term}/lineage?version=latest'
    try:
        result = requests.get(lookup_url.format(input_term=input_type.replace(':', '%3A')))
        if result.status_code != 200:
            print(input_type, result)
            return []
        return result.json()
    except Exception as ex:
        print(ex)
        return []


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file', type=str,
                        default='../data/comparison/diff_predicted_types.csv',
                        help='input file of different prediction types from two models')
    parser.add_argument('--column1', type=str,
                        default='Biomegatron predicted type',
                        help='the first column name to compare type with')
    parser.add_argument('--column2', type=str,
                        default='Sapbert predicted type',
                        help='the second column name to compare type with')

    args = parser.parse_args()
    input_file = args.input_file
    column1 = args.column1
    column2 = args.column2

    df = pd.read_csv(input_file)
    print(df.shape)
    df_type_unique = df[column1].unique()
    hierarchy_dict = {}
    for elem in df_type_unique:
        if elem != 'biolink:NamedThing':
            rel_ary = get_hierarchy_relation(elem)
            if rel_ary:
                hierarchy_dict[elem] = rel_ary
    print(hierarchy_dict)
    df['hierarchical_relation'] = df.apply(
        lambda row: True if row[column1] == 'biolink:NamedThing' or row[column2] == 'biolink:NamedThing' or
                            row[column1] in hierarchy_dict and row[column2] in hierarchy_dict[row[column1]] else False,
        axis=1)
    df_true = df[df['hierarchical_relation'] == True]
    df_false = df[df['hierarchical_relation'] == False]
    base_input_name = os.path.splitext(input_file)[0]
    df_true.to_csv(f'{base_input_name}_true.csv', index=False)
    df_false.to_csv(f'{base_input_name}_false.csv', index=False)
