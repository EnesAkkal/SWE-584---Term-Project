import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)
        self.features = None
        self.target = None
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.rf_model = RandomForestClassifier(n_estimators=30, random_state=42)
        self.lr_model = LogisticRegression(random_state=42, max_iter=50)
        print(self.data.columns)

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
            'satisfaction': lambda x: 1 if x == 'satisfied' else 0
        }
        for col, func in transformations.items():
            self.data[col] = self.data[col].apply(func)
        
        class_dummies = pd.get_dummies(self.data['Class'], prefix='Class')
        self.data = pd.concat([self.data.drop('Class', axis=1), class_dummies], axis=1)
        
        self.data['Arrival Delay in Minutes'].fillna(self.data['Arrival Delay in Minutes'].median(), inplace=True)
        self.data.drop(['Gender', 'Departure Delay in Minutes', 'Food and drink'], axis=1, inplace=True)
        
        self.features = self.data.drop('satisfaction', axis=1)
        self.target = self.data['satisfaction']
        print("First 5 rows of the processed data:")
        print(self.data.head())
        return self.data

    def combine_class_features(self):
        # Combine class features
        if 'Class_Business' in self.data.columns and 'Class_Eco' in self.data.columns and 'Class_Eco Plus' in self.data.columns:
            self.data['Class'] = (self.data['Class_Business'] * 2 + self.data['Class_Eco'] + self.data['Class_Eco Plus']) / 4
            self.data.drop(['Class_Business', 'Class_Eco', 'Class_Eco Plus'], axis=1, inplace=True)
        else:
            print("Class columns are not found in the data.")


    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.features, self.target, test_size=0.5, random_state=42
        )
        scaler = StandardScaler()
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_test = scaler.transform(self.X_test)

    def train_model(self):
        self.rf_model.fit(self.X_train, self.y_train)
        self.lr_model.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        rf_pred = self.rf_model.predict(self.X_test)
        lr_pred = self.lr_model.predict(self.X_test)

        # Print classification reports
        print("Random Forest Accuracy:", accuracy_score(self.y_test, rf_pred))
        print("Random Forest Classification Report:")
        print(classification_report(self.y_test, rf_pred))

        lr_pred = self.lr_model.predict(self.X_test)
        print("Logistic Regression Accuracy:", accuracy_score(self.y_test, lr_pred))
        print("Logistic Regression Classification Report:")
        print(classification_report(self.y_test, lr_pred))

        # Confusion Matrices
        cm_rf = confusion_matrix(self.y_test, rf_pred)
        cm_lr = confusion_matrix(self.y_test, lr_pred)

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(cm_rf, annot=True, fmt="d", ax=ax[0], cmap='Blues')
        ax[0].set_title('Random Forest Confusion Matrix')
        ax[0].set_xlabel('Predicted Labels')
        ax[0].set_ylabel('True Labels')

        sns.heatmap(cm_lr, annot=True, fmt="d", ax=ax[1], cmap='Blues')
        ax[1].set_title('Logistic Regression Confusion Matrix')
        ax[1].set_xlabel('Predicted Labels')
        ax[1].set_ylabel('True Labels')
        
        plt.show()

    def plot_feature_importance(self):
        importances = self.rf_model.feature_importances_
        feature_names = self.features.columns
        feature_importance_dict = dict(zip(feature_names, importances))
        importances_df = pd.DataFrame(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True),
                                      columns=['Feature', 'Importance'])
        plt.figure(figsize=(12, 10))
        sns.barplot(x='Importance', y='Feature', data=importances_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.subplots_adjust(left=0.3)
        plt.show()

    def combine_class_features(self):
        if 'Class_Business' in self.data.columns and 'Class_Eco' in self.data.columns and 'Class_Eco Plus' in self.data.columns:
            self.data['Class'] = (self.data['Class_Business'] * 2 + self.data['Class_Eco'] + self.data['Class_Eco Plus']) / 4
            self.data.drop(['Class_Business', 'Class_Eco', 'Class_Eco Plus'], axis=1, inplace=True)

    def plot_correlation_matrix(self):
        plt.figure(figsize=(20, 16))
        correlation_matrix = self.data.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
        plt.title('Correlation Matrix Heatmap', fontsize=18)
        plt.xticks(rotation=90, fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.tight_layout()
        plt.show()

    def feature_importance(self):
        importances = self.rf_model.feature_importances_
        feature_names = self.features.columns
        feature_importance_dict = dict(zip(feature_names, importances))

        class_importance = sum(importance for name, importance in feature_importance_dict.items() if 'Class_' in name)
        feature_importance_dict['Class'] = class_importance
        
        for key in list(feature_importance_dict.keys()):
            if 'Class_' in key:
                del feature_importance_dict[key]

        importances_df = pd.DataFrame(sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True),
        columns=['Feature', 'Importance'])
        return importances_df

    def plot_feature_importance(self):
        importances_df = self.feature_importance()
        plt.figure(figsize=(12, 10))  
        sns.barplot(x='Importance', y='Feature', data=importances_df)
        plt.title('Feature Importance')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.subplots_adjust(left=0.3)  
        plt.show()

    def plot_distributions(self):
        plt.figure(figsize=(18, 5))
        plt.subplot(1, 3, 1)
        sns.histplot(self.data['Online boarding'], kde=True, color='blue')
        plt.title('Distribution of Online Boarding')
        plt.subplot(1, 3, 2)
        sns.histplot(self.data['Inflight wifi service'], kde=True, color='green')
        plt.title('Distribution of Inflight Wifi Service')
        plt.subplot(1, 3, 3)
    
        if 'Class' in self.data.columns:
            sns.histplot(self.data['Class'], bins=len(self.data['Class'].unique()), color='orange')
        else:  
            total_class = self.data[[col for col in self.data.columns if 'Class_' in col]].idxmax(axis=1)
            sns.histplot(total_class, color='orange')
        plt.title('Distribution of Class')

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    processor = DataProcessor('Dataset/data.csv')
    processor.process_data()
    processor.split_data()
    processor.train_model()
    processor.plot_feature_importance()
    processor.plot_distributions()
    processor.combine_class_features()  
    processor.plot_correlation_matrix()
    processor.evaluate_model()
