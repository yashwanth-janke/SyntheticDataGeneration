import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(df, target_column='class'):
    df = df.dropna()  # Handle missing values
    df = pd.get_dummies(df)  # Encode categorical variables
    X = df.drop([target_column], axis=1)
    y = df[target_column]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)
    return X_train, X_test
