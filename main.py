import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.features = None
        self.target = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lr_model = LogisticRegression(random_state=42, max_iter=500)

    @staticmethod
    def transform_gender(x):
        return 1 if x == 'Female' else 0 if x == 'Male' else -1

    @staticmethod
    def transform_customer_type(x):
        return 1 if x == 'Loyal Customer' else 0 if x == 'disloyal Customer' else -1

    def process_data(self):
        self.data = self.data.drop(['Unnamed: 0', 'id'], axis=1)
        transformations = {
            'Gender': self.transform_gender,
            'Customer Type': self.transform_customer_type,
            'Type of Travel': lambda x: 1 if x == 'Business travel' else 0,
            'Class': lambda x: {'Business': 2, 'Eco Plus': 1, 'Eco': 0}.get(x, -1),
            'satisfaction': lambda x: 1 if x == 'satisfied' else 0
        }
        for col, func in transformations.items():
            self.data[col] = self.data[col].apply(func)
        self.data['Arrival Delay in Minutes'].fillna(self.data['Arrival Delay in Minutes'].median(), inplace=True)
        self.features = self.data.drop('satisfaction', axis=1)
        self.target = self.data['satisfaction']
        return self.data

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=0.5, random_state=42
        )
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):
        # Train Random Forest
        self.rf_model.fit(self.X_train, self.y_train)
        print("Random Forest model trained successfully.")

        # Train Logistic Regression
        self.lr_model.fit(self.X_train, self.y_train)
        print("Logistic Regression model trained successfully.")

    def evaluate_model(self):
        # Evaluate Random Forest
        rf_pred = self.rf_model.predict(self.X_test)
        print("Random Forest Accuracy:", accuracy_score(self.y_test, rf_pred))
        print("Random Forest Classification Report:")
        print(classification_report(self.y_test, rf_pred))

        # Evaluate Logistic Regression
        lr_pred = self.lr_model.predict(self.X_test)
        print("Logistic Regression Accuracy:", accuracy_score(self.y_test, lr_pred))
        print("Logistic Regression Classification Report:")
        print(classification_report(self.y_test, lr_pred))
    
    def feature_importance(self):
        importances = self.rf_model.feature_importances_
        feature_names = self.features.columns
        feature_importance_dict = dict(zip(feature_names, importances))
        importances_df = pd.DataFrame(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True),
        columns=['Feature', 'Importance'])
        
        return importances_df

    def plot_feature_importance(self):
        importances_df = self.feature_importance()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=importances_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.show()

# Usage
if __name__ == "__main__":
    processor = DataProcessor('Dataset/train.csv')
    processor.process_data()
    processor.split_data()
    processor.train_model()
    processor.evaluate_model()
    processor.plot_feature_importance()
