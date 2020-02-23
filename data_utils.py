import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_intention():
    train = pd.read_csv("data/online_shoppers_intention.csv")
    train.drop_duplicates(inplace=True)
    train.dropna(inplace=True)

    train.OperatingSystems = train.OperatingSystems.astype('category')
    train.Browser = train.Browser.astype('category')
    train.Region = train.Region.astype('category')
    train.TrafficType = train.TrafficType.astype('category')

    # Normalize numeric columns
    numeric_cols = train.select_dtypes(include=['float64']).columns.tolist()
    train[numeric_cols] = train[numeric_cols].apply(lambda x: (x - x.mean())/x.std())
    

    # numerical_features = [col for col in train.columns if train.get
    train['Weekend'] = train['Weekend'].astype('category')
    train = pd.get_dummies(train, drop_first=True)
    target = train.Revenue.astype('int8')



    train.drop("Revenue", axis=1, inplace=True)
    return train, target

X, y = load_intention()
