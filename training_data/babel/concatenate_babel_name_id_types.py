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
                        default='/projects/ner/software/sapbert/sapbert/data/babel/2024_v2/updated_mapping',
                        help='input file directory to concatenate with input_file_list')
    parser.add_argument('--input_file_list', type=list, default=[
        'AnatomicalEntity.txt', 'GrossAnatomicalStructure.txt', 'Cell.txt',
        'CellularComponent.txt', 'BiologicalProcess.txt', 'Pathway.txt', 'MolecularActivity.txt',
        'PhenotypicFeature.txt', 'Disease.txt', 'umls.txt', 'OrganismTaxon.txt',
        'Gene.txt', 'Protein.txt', 'DrugChemicalConflated.txt', 'GeneFamily.txt'
    ], help='input file list to process')

    args = parser.parse_args()
    input_file_dir = args.input_file_dir
    input_file_list = args.input_file_list
    id_type_dfs = []
    name_id_dfs = []

    for f in input_file_list:
        base_f = os.path.splitext(f)[0]
        print(f'processing {f}', flush=True)
        name_id_df = pd.read_csv(os.path.join(input_file_dir, f'{base_f}_name_ids.csv'))
        name_id_df['Name'] = name_id_df['Name'].str.strip()
        name_id_df['ID'] = name_id_df['ID'].str.strip()
        name_id_dfs.append(name_id_df)
        # create id-type mapping data frame
        id_type_df = pd.read_csv(os.path.join(input_file_dir, f'{base_f}_id_types.csv'))
        id_type_df['id'] = id_type_df['id'].str.strip()
        id_type_df['type'] = id_type_df['type'].str.strip()
        id_type_dfs.append(id_type_df)


    concat_id_type_df = pd.concat(id_type_dfs).drop_duplicates()
    # combine multiple types mapped from the same id into a set using pivot table
    df_id_type_pivot = pd.pivot_table(concat_id_type_df, values=['type'], index='id', aggfunc={'type': list})
    df_id_type_pivot.to_csv(os.path.join(input_file_dir, 'id_types.csv'), index=True)
    ni_df = pd.concat(name_id_dfs).drop_duplicates()
    df_ni_pivot = pd.pivot_table(ni_df, values=['ID'], index='Name', aggfunc={'ID': list})
    df_ni_pivot.to_csv(os.path.join(input_file_dir, 'name_ids.csv'), index=True)