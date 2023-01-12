import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file_dir', type=str,
                        default='/projects/babel/sapbert-training/2022dec2-2-lc/compendia/',
                        help='input file directory to concatenate with input_file_list')
    parser.add_argument('--input_file_list', type=list, default=[
       'AnatomicalEntity.txt', 'GrossAnatomicalStructure.txt', 'ComplexMolecularMixture.txt',
        'ChemicalMixture.txt', 'Polypeptide.txt', 'Cell.txt',
        'CellularComponent.txt', 'BiologicalProcess.txt', 'Pathway.txt', 'MolecularActivity.txt',
        'PhenotypicFeature.txt', 'Disease.txt', 'umls.txt', 'OrganismTaxon.txt', 'ChemicalEntity.txt',
        'MolecularMixture.txt', 'Gene.txt', 'SmallMolecule.txt', 'Protein.txt'
    ], help='input file list to process')
    parser.add_argument('--concatenate_all', action="store_true")
    parser.add_argument('--output_path', type=str,
                        default='/projects/ner/software/sapbert/sapbert/data/babel/mapping',
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
        df = pd.read_csv(os.path.join(input_file_dir, f), sep='\|\|', header=None)
        df.columns = ['type', 'id', 'name1', 'name2']
        # create id-type mapping data frame
        id_type_df = df.groupby(['id', 'type']).size().reset_index().rename(columns={0:'count'})
        id_type_df.drop(columns=['count'], inplace=True)
        if concatenate_all:
            id_type_dfs.append(id_type_df)
        else:
            id_type_df.to_csv(os.path.join(output_path, f'{base_f}_id_types.csv'), index=False)
        df1 = df.groupby(['name1', 'id']).size().reset_index().rename(
            columns={0:'count', 'name1': 'Name', 'id': 'ID'})
        df2 = df.groupby(['name2', 'id']).size().reset_index().rename(
            columns={0: 'count', 'name2': 'Name', 'id': 'ID'})
        name_id_df = pd.concat([df1, df2]).drop(columns=['count']).drop_duplicates()
        if concatenate_all:
            name_id_dfs.append(name_id_df)
        else:
            name_id_df.to_csv(os.path.join(output_path, f'{base_f}_name_ids.csv'), index=False)

    if concatenate_all:
        pd.concat(id_type_dfs).drop_duplicates().to_csv(os.path.join(output_path, 'id_types.csv'), index=False)
        pd.concat(name_id_dfs).drop_duplicates().to_csv(os.path.join(output_path, 'name_ids.csv'), index=False)
