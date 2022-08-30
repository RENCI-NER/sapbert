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
    parser.add_argument('--column_1_name', type=str, default='Labelled Category', help='first column to compare')
    parser.add_argument('--column_2_name', type=str, default='Predicted Category', help='second column to compare')
    parser.add_argument('--list_comparision', type=bool, default=False, help='whether to do list comparison or not')

    args = parser.parse_args()
    input_file = args.input_file
    column_1_name = args.column_1_name
    column_2_name = args.column_2_name
    list_comarison = args.list_comparision

    df = pd.read_csv(input_file, usecols=[column_1_name, column_2_name], dtype=str)
    if list_comarison:
        df['Prediction_agreement'] = df.apply(lambda x: x[column_2_name] in x[column_1_name], axis=1)
    else:
        df['Prediction_agreement'] = df.apply(lambda x: x[column_1_name] == x[column_2_name], axis=1)

    agree_df = df[df.Prediction_agreement == True]
    disagree_df = df[df.Prediction_agreement == False]
    if not list_comarison:
        # get percentage of empty column_1_name and column_2_name
        col_1_null = df[column_1_name].isna()
        col_2_null = df[column_2_name].isna()
        print(f'{column_1_name} null percentage: {len(col_1_null.loc[lambda x: x == True])/len(df)}')
        print(f'{column_2_name} null percentage: {len(col_2_null.loc[lambda x: x == True])/len(df)}')
    print(f'agreement percentage: {len(agree_df)/len(df)}')
