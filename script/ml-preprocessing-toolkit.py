import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


class preprocess:
    @staticmethod
    def clean_data(data):
        # Drop duplicates
        data = data.drop_duplicates()
        # Strip whitespace from string columns
        for col in data.select_dtypes(include=['object']).columns:
            data[col] = data[col].str.strip()
        # Fill infinite values with NaN
        data = data.replace([float('inf'), float('-inf')], pd.NA)
        # Drop columns with all NaN values
        data = data.dropna(axis=1, how='all')
        # Optionally, drop rows with all NaN values
        data = data.dropna(axis=0, how='all')
        return data

    @staticmethod
    def preprocess_data(data):
        numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = data.select_dtypes(include=['object']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        X_processed = preprocessor.fit_transform(data)
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names = preprocessor.get_feature_names_out()
        else:
            num_names = numeric_features
            cat_names = preprocessor.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_features)
            feature_names = list(num_names) + list(cat_names)
        return X_processed, feature_names, preprocessor

    @staticmethod
    def load_data(filepath):
        data = pd.read_csv(filepath)
        data = preprocess.clean_data(data)
        return data

    @staticmethod
    def plot_corr_map(X_processed, feature_names):
        X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
        corr_processed = X_processed_df.corr()
        plt.figure(figsize=(12, 8))
        sns.heatmap(corr_processed, annot=False, cmap='coolwarm', linewidths=0.5)
        plt.title("Correlation Heatmap (Preprocessed Data)")
        plt.show()
    @staticmethod
    def add_data(data,count):
        n_to_add = count
        # Randomly sample and append
        new_rows = data.sample(n=n_to_add, replace=True, random_state=42)
        data_augmented = pd.concat([data, new_rows], ignore_index=True)
        print("After:", data_augmented.shape)
        data_augmented.to_csv('../datasets/student_new.csv', index=False)


# --- Main script logic ---
data = preprocess.load_data('../datasets/student_new.csv')
print(data.shape)

add_data=preprocess.add_data(data,1000)
X_processed, feature_names, preprocessor = preprocess.preprocess_data(data)
preprocess.plot_corr_map(X_processed, feature_names)