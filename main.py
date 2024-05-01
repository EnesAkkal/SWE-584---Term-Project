import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
file_path = 'Dataset/train.csv'
data = pd.read_csv(file_path)
print(data.head())  # This will print the first 5 rows of your DataFrame


import pandas as pd

class DataProcessor:
    def __init__(self, file_path):
        self.data = pd.read_csv(file_path)

    @staticmethod
    def transform_gender(x):
        if x == 'Female':
            return 1
        elif x == 'Male':
            return 0
        else:
            return -1

    @staticmethod
    def transform_customer_type(x):
        if x == 'Loyal Customer':
            return 1
        elif x == 'disloyal Customer':
            return 0
        else:
            return -1

    @staticmethod
    def transform_travel_type(x):
        if x == 'Business travel':
            return 1
        elif x == 'Personal Travel':
            return 0
        else:
            return -1

    @staticmethod
    def transform_class(x):
        if x == 'Business':
            return 2
        elif x == 'Eco Plus':
            return 1
        elif x == 'Eco':
            return 0
        else:
            return -1

    @staticmethod
    def transform_satisfaction(x):
        if x == 'satisfied':
            return 1
        elif x == 'neutral or dissatisfied':
            return 0
        else:
            return -1

    def process_data(self):
        self.data = self.data.drop(['Unnamed: 0', 'id'], axis=1)
        self.data['Gender'] = self.data['Gender'].apply(self.transform_gender)
        self.data['Customer Type'] = self.data['Customer Type'].apply(self.transform_customer_type)
        self.data['Type of Travel'] = self.data['Type of Travel'].apply(self.transform_travel_type)
        self.data['Class'] = self.data['Class'].apply(self.transform_class)
        self.data['satisfaction'] = self.data['satisfaction'].apply(self.transform_satisfaction)
        self.data['Arrival Delay in Minutes'].fillna(self.data['Arrival Delay in Minutes'].median(), inplace=True)

        return self.data

# Usage
if __name__ == "__main__":
    train_processor = DataProcessor('Dataset/train.csv')

    train_data = train_processor.process_data()

    print(train_data.head())

def split_data(self):
        features = ['Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
                    'Flight Distance', 'Inflight wifi service',
                    'Departure/Arrival time convenient', 'Ease of Online booking',
                    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
                    'Inflight entertainment', 'On-board service', 'Leg room service',
                    'Baggage handling', 'Checkin service', 'Inflight service',
                    'Cleanliness', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
        target = 'satisfaction'
        
        X_train = self.train[features]
        y_train = self.train[target]
        X_test = self.test[features]
        y_test = self.test[target]

        # Normalize Features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)  # Use transform here to avoid data leakage

        return X_train_scaled, y_train, X_test_scaled, y_test


