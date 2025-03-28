import pandas as pd
import os
import argparse
import numpy as np


def concatenate_all_for_mapping(name_ids, id_types, out_dir):
    # Concatenate and aggregate ID-type mappings
    concat_id_type_df = pd.concat(id_types).groupby(['id', 'type'])['count'].sum().reset_index()

    # Filter out biolink:OrganismTaxon where it co-occurs with another type
    id_counts = concat_id_type_df.groupby('id').size()
    ids_with_multiple_types = id_counts[id_counts > 1].index
    multi_type_df = concat_id_type_df[concat_id_type_df['id'].isin(ids_with_multiple_types)]
    single_type_df = concat_id_type_df[~concat_id_type_df['id'].isin(ids_with_multiple_types)]

    # Separate cases where biolink:OrganismTaxon is one of the types
    organism_taxon_df = multi_type_df[multi_type_df['type'] == 'biolink:OrganismTaxon']
    non_organism_df = multi_type_df[multi_type_df['type'] != 'biolink:OrganismTaxon']

    non_organism_counts = non_organism_df.groupby('id').size()
    exception_ids = non_organism_counts[non_organism_counts > 1].index
    if not exception_ids.empty:
        exception_df = non_organism_df[non_organism_df['id'].isin(exception_ids)]
        print("Warning: Found IDs with multiple non-biolink:OrganismTaxon types (rule violation):")
        print(exception_df[['id', 'type', 'count']].to_string(index=False))
        exception_df.to_csv(os.path.join(out_dir, 'id_type_exceptions.csv'), index=False)

    # keeping only non-biolink:OrganismTaxon types for IDs with 2 types
    resolved_multi_df = non_organism_df.groupby('id').agg({'type': 'first'}).reset_index()

    # Combine resolved multi-type IDs with single-type IDs
    resolved_id_type_df = pd.concat([single_type_df[['id', 'type']], resolved_multi_df]).drop_duplicates()

    # Merge with name-ID mapping
    concat_name_id_df = pd.concat(name_ids).drop_duplicates()
    merged_df = pd.merge(concat_name_id_df, resolved_id_type_df, on='id', how='left')
    print(f'before dropna(), merged_df.shape: {merged_df.shape}')
    merged_df = merged_df[['name', 'id', 'type']].dropna()  # Drop rows where type is missing
    print(f'after dropna(), merged_df.shape: {merged_df.shape}')
    # Pivot to create name-to-id-to-type mapping with lists
    final_df = merged_df.groupby('name').agg({'id': list, 'type': list}).reset_index()

    # Save the final mapping
    final_df.to_csv(os.path.join(out_dir, 'name_id_type_mapping.csv'), index=False)


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
    id_type_dfs = []
    name_id_dfs = []
    unique_types = []
    for f in input_file_list:
        base_f = os.path.splitext(f)[0]
        print(f'processing {f}', flush=True)
        df = pd.read_csv(os.path.join(input_file_dir, f), sep='\|\|', header=None, engine='python')
        df.columns = ['type', 'id', 'name', 'name1', 'name2']
        # create id-type mapping data frame
        unique_types.append(df['type'].unique())
        id_type_df = df.groupby(['id', 'type']).size().reset_index().rename(columns={0: 'count'})
        if concatenate_all:
            # keep the count column since it is used later
            id_type_dfs.append(id_type_df)
        id_type_df.to_csv(os.path.join(output_path, f'{base_f}_id_types.csv'), index=False)
        name_id_df = df.groupby(['name', 'id']).size().reset_index().rename(columns={0: 'count'})
        name_id_df = name_id_df.drop(columns=['count'])
        if concatenate_all:
            name_id_dfs.append(name_id_df)
        name_id_df.to_csv(os.path.join(output_path, f'{base_f}_name_ids.csv'), index=False)

    if unique_types:
        print(f'unique biolink types: {np.unique(np.concatenate(unique_types))}', flush=True)
    else:
        print("No unique types found (no files processed)", flush=True)

    if concatenate_all:
        concatenate_all_for_mapping(name_id_dfs, id_type_dfs, output_path)
