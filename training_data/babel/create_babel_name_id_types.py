import pandas as pd
import os
import argparse


def map_ids_to_types(id_list, df_id_types):
    ids_to_types = {}
    for id in id_list:
        ids_to_types[id] = sum(df_id_types[df_id_types.id==id]['type'].tolist(), [])
    return ids_to_types


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file_dir', type=str,
                        default='/projects/babel/babel-outputs/2024jan9/sapbert-training-data',
                        help='input file directory to concatenate with input_file_list')
    parser.add_argument('--input_file_list', type=list, default=[
        'AnatomicalEntity.txt', 'GrossAnatomicalStructure.txt', 'Cell.txt',
        'CellularComponent.txt', 'BiologicalProcess.txt', 'Pathway.txt', 'MolecularActivity.txt',
        'PhenotypicFeature.txt', 'Disease.txt', 'umls.txt', 'OrganismTaxon.txt',
        'Gene.txt', 'Protein.txt', 'DrugChemicalConflated.txt', 'GeneFamily.txt'
    ], help='input file list to process')
    parser.add_argument('--concatenate_all', action="store_true")
    parser.add_argument('--output_path', type=str,
                        default='/projects/ner/software/sapbert/sapbert/data/babel/2024_v2/updated_mapping',
                        help='output path to write name-id pairs and id-type pairs')

    args = parser.parse_args()
    output_path = args.output_path
    concatenate_all = args.concatenate_all
    input_file_dir = args.input_file_dir
    input_file_list = args.input_file_list
    if concatenate_all:
        id_type_dfs = []
        name_id_dfs = []
    for f in input_file_list:
        base_f = os.path.splitext(f)[0]
        print(f'processing {f}', flush=True)
        df = pd.read_csv(os.path.join(input_file_dir, f), sep='\|\|', header=None, engine='python')
        df.columns = ['type', 'id', 'name', 'name1', 'name2']
        # create id-type mapping data frame
        id_type_df = df.groupby(['id', 'type']).size().reset_index().rename(columns={0: 'count'})
        id_type_df.drop(columns=['count'], inplace=True)
        if concatenate_all:
            id_type_dfs.append(id_type_df)
        else:
            id_type_df.to_csv(os.path.join(output_path, f'{base_f}_id_types.csv'), index=False)
        name_id_df = df.groupby(['name', 'id']).size().reset_index().rename(
            columns={0: 'count', 'name': 'Name', 'id': 'ID'})
        name_id_df = name_id_df.drop(columns=['count'])
        if concatenate_all:
            name_id_dfs.append(name_id_df)
        else:
            name_id_df.to_csv(os.path.join(output_path, f'{base_f}_name_ids.csv'), index=False)

    if concatenate_all:
        concat_id_type_df = pd.concat(id_type_dfs).drop_duplicates()
        # combine multiple types mapped from the same id into a set using pivot table
        df_id_type_pivot = pd.pivot_table(concat_id_type_df, values=['type'], index='id', aggfunc={'type': list})
        df_id_type_pivot.to_csv(os.path.join(output_path, 'id_types.csv'), index=True)
        ni_df = pd.concat(name_id_dfs).drop_duplicates()
        df_ni_pivot = pd.pivot_table(ni_df, values=['ID'], index='Name', aggfunc={'ID': list})
        df_ni_pivot.to_csv(os.path.join(output_path, 'name_ids.csv'), index=True)
        # note when reading dataframe back, ID column will be of string type, which can be converted back to list
        # by using df_ni_pivot['ID'] = df_ni_pivot['ID'].map(lambda d: ast.literal_eval(d))
        # df_id_type_pivot['type'] = df_id_type_pivot['type'].map(lambda d: ast.literal_eval(d))
        # note the map_ids_to_types() applied to each row could take a very long time
        # df_ni_pivot['id_type'] = df_ni_pivot.apply(lambda row: map_ids_to_types(row.ID, df_id_type_pivot), axis=1)
        # df_ni_pivot.drop(columns=['ID'])
        # df_ni_pivot.to_csv(os.path.join(output_path, 'name_id_types.csv'), index=True)
