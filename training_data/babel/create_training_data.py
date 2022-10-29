import pandas as pd
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    parser.add_argument('--input_file_name_1', type=str,
                        default='/projects/babel/sapbert-training/2022oct13-lc/compendia/AnatomicalEntity.txt',
                        help='the first input file to drop the first column and combine with the second input file')
    parser.add_argument('--input_file_name_2', type=str,
                        default='/projects/babel/sapbert-training/2022oct13-lc/compendia/GrossAnatomicalStructure.txt',
                        help='the second input file to drop the first column and combine with the first input file')
    parser.add_argument('--input_file_dir', type=str,
                        default='/projects/babel/sapbert-training/2022oct13-lc/compendia/',
                        help='input file directory to concatenate with input_file_list')
    parser.add_argument('--input_file_list', type=list, default=[
       'ComplexMolecularMixture.txt', 'ChemicalMixture.txt', 'Polypeptide.txt', 'Cell.txt',
        'CellularComponent.txt', 'BiologicalProcess.txt', 'Pathway.txt', 'MolecularActivity.txt',
        'PhenotypicFeature.txt', 'Disease.txt', 'umls.txt', 'OrganismTaxon.txt', 'ChemicalEntity.txt',
        'MolecularMixture.txt', 'Gene.txt', 'SmallMolecule.txt', 'Protein.txt'
    ], help='input file list to process')
    parser.add_argument('--remove_last_char_in_last_column', action="store_true")
    parser.add_argument('--output_file', type=str,
                        default='/projects/ner/software/sapbert/sapbert/data/babel/anatomy.txt',
                        help='output file to combine two input files while dropping the first column to create '
                             'initial sapbert training data')

    args = parser.parse_args()
    input_file_name_1 = args.input_file_name_1
    input_file_name_2 = args.input_file_name_2
    output_file = args.output_file
    remove_last_char_in_last_column = args.remove_last_char_in_last_column
    input_file_dir = args.input_file_dir
    input_file_list = args.input_file_list

    if not input_file_list:
        df1 = pd.read_csv(input_file_name_1, sep='\|\|', header=None, usecols=[1, 2, 3])
        df2 = pd.read_csv(input_file_name_2, sep='\|\|', header=None, usecols=[1, 2, 3])
        if remove_last_char_in_last_column:
            df1[3] = df1[3].str[:-1]
            df2[3] = df2[3].str[:-1]

        df = pd.concat([df1, df2], ignore_index=True)
        # since pandas does not support multiple character separator, cannot directly use to_csv to
        # write data frame to csv with separator ||
        row_series = df[df.columns].astype(str).apply(lambda x: '||'.join(x), axis=1)
        row_series.to_csv(output_file, header=False, sep='\t', index=False)
    else:
        for f in input_file_list:
            df = pd.read_csv(os.path.join(input_file_dir, f), sep='\|\|', header=None, usecols=[1, 2, 3])
            if remove_last_char_in_last_column:
                df[3] = df[3].str[:-1]
            # since pandas does not support multiple character separator, cannot directly use to_csv to
            # write data frame to csv with separator ||
            row_series = df[df.columns].astype(str).apply(lambda x: '||'.join(x), axis=1)
            row_series.to_csv(os.path.join(os.path.dirname(output_file), f),
                              header=False, sep='\t', index=False)
