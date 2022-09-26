import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process arguments.')
    # parser.add_argument('--input_file', type=str, default='../data/pubmed_split_11_prediction_output.csv',
    #                     help='input file for comparison between prediction columns')
    # parser.add_argument('--column_1_name', type=str, default='Labelled MESH ID', help='first column to compare')
    # parser.add_argument('--column_2_name', type=str, default='Predicted MESH ID', help='second column to compare')
    parser.add_argument('--input_file', type=str, default='../data/annotated_17687_output_biolink.csv',
                        help='input file for comparison between prediction columns')
    parser.add_argument('--input_text_column_name', type=str, default='Span text', help='input text column name')
    parser.add_argument('--column_1_name', type=str, default='Biomegatron predicted type',
                        help='first column to compare')
    parser.add_argument('--column_2_name', type=str, default='Sapbert predicted type', help='second column to compare')
    parser.add_argument('--list_comparision', type=bool, default=False, help='whether to do list comparison or not')
    parser.add_argument('--diff_output_file', type=str, default='../data/comparison/diff_predicted_types.csv',
                        help='output file name for different prediction breakdown')
    parser.add_argument('--same_null_output_file', type=str, default='../data/comparison/same_null_predicted_types.csv',
                        help='output file name for same null predictions')
    parser.add_argument('--same_not_null_output_file', type=str,
                        default='../data/comparison/same_not_null_predicted_types.csv',
                        help='output file name for same not null predictions')

    args = parser.parse_args()
    input_file = args.input_file
    input_text_column_name = args.input_text_column_name
    column_1_name = args.column_1_name
    column_2_name = args.column_2_name
    list_comarison = args.list_comparision
    diff_output_file = args.diff_output_file
    same_null_output_file = args.same_null_output_file
    same_not_null_output_file = args.same_not_null_output_file

    df = pd.read_csv(input_file, usecols=[input_text_column_name, column_1_name, column_2_name], dtype=str)
    print(f'before droping duplicates: {df.shape}')
    df.drop_duplicates(inplace=True)
    print(f'after droping duplicates: {df.shape}')
    df = df[df['Span text'].isna() == False]
    df[column_2_name].fillna('biolink:NamedThing', inplace=True)
    if list_comarison:
        df['Prediction_agreement'] = df.apply(lambda x: x[column_2_name] in x[column_1_name], axis=1)
    else:
        df['Prediction_agreement'] = df.apply(lambda x: x[column_1_name] == x[column_2_name], axis=1)

    agree_df = df[df.Prediction_agreement == True]
    disagree_df = df[df.Prediction_agreement == False]
    if not list_comarison:
        # get percentage of empty column_1_name and column_2_name
        col_1_null = df[df[column_1_name] == 'biolink:NamedThing']
        col_2_null = df[df[column_2_name] == 'biolink:NamedThing']
        col_12_null = df[(df[column_1_name] == 'biolink:NamedThing') & (df[column_2_name] == 'biolink:NamedThing')]
        col_12_same_not_null = df[(df[column_1_name] == df[column_2_name]) &
                                  (df[column_1_name] != 'biolink:NamedThing') &
                                  (df[column_2_name] != 'biolink:NamedThing')]
        print(f'{column_1_name} biolink:NamedThing percentage: {len(col_1_null) / len(df)}')
        print(f'{column_2_name} biolink:NamedThing percentage: {len(col_2_null) / len(df)}')
        print(f'Both {column_1_name} and {column_2_name} biolink:NamedThing percentage: {len(col_12_null) / len(df)}')
        diff_df = df[df['Prediction_agreement'] == False].sort_values(by=[column_1_name, column_2_name])
        diff_df.drop(columns=['Prediction_agreement'], inplace=True)
        diff_df.to_csv(diff_output_file, index=False)
        col_12_null.drop(columns=['Prediction_agreement']).sort_values(by=[input_text_column_name]).to_csv(
            same_null_output_file, index=False)
        col_12_same_not_null.drop(columns=['Prediction_agreement']).sort_values(by=[input_text_column_name]).to_csv(
            same_not_null_output_file, index=False)

    print(f'agreement percentage: {len(agree_df) / len(df)}, disagreement percentage: {len(disagree_df) / len(df)}')
