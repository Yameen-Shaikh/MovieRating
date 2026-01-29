import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

def preprocess_data(df):
    # Drop rows with missing values for simplicity
    df.dropna(inplace=True)

    # Clean 'Year' column
    df['Year'] = df['Year'].str.extract(r'(\d{4})').astype(int)

    # Clean 'Duration' column
    df['Duration'] = df['Duration'].str.replace(' min', '').astype(int)

    # Clean 'Votes' column
    df['Votes'] = df['Votes'].str.replace(',', '').astype(int)

    # Encode 'Genre'
    df['Genre'] = df['Genre'].str.split(', ')
    mlb = MultiLabelBinarizer()
    genre_encoded = mlb.fit_transform(df['Genre'])
    genre_df = pd.DataFrame(genre_encoded, columns=mlb.classes_, index=df.index)
    df = pd.concat([df.drop('Genre', axis=1), genre_df], axis=1)


    le = LabelEncoder()
    # Encode 'Director'
    df['Director'] = le.fit_transform(df['Director'])

    # Encode 'Actor 1'
    df['Actor 1'] = le.fit_transform(df['Actor 1'])

    # Encode 'Actor 2'
    df['Actor 2'] = le.fit_transform(df['Actor 2'])

    # Encode 'Actor 3'
    df['Actor 3'] = le.fit_transform(df['Actor 3'])

    return df

if __name__ == '__main__':
    df = pd.read_csv(r"C:\Users\Sumay\Desktop\Yameen\Codesoft\DataScience\IMDb Movies India.csv\IMDb Movies India.csv", encoding='latin1')
    df_cleaned = preprocess_data(df)
    print(df_cleaned.head())
    print(df_cleaned.info())