import pandas as pd #Specifies the number of trees in the forest
from sklearn.model_selection import train_test_split #split the dataset into training and testing sets
from sklearn.ensemble import RandomForestClassifier # ML Model for classification
from sklearn.preprocessing import LabelEncoder  #encoding categorical data into numerical format
from sklearn.metrics import classification_report
import joblib  #saving the trained model and encoders for future use

# Load the dataset
file_path = 'Crop_production.csv'  # Path to dataset
data = pd.read_csv(file_path)

# Verify column names in the dataset
print("Dataset columns:", data.columns)

# Step 1: Data Preprocessing
data.rename(columns={
    'State': 'State_Name',
    'Nitrogen': 'N',
    'Phosphorus': 'P',
    'Potassium': 'K',
    'pH_value': 'pH',
    'Rain_mm': 'Rainfall',
    'Temp_C': 'Temperature'
}, inplace=True)

# Encode categorical columns
label_encoder_state = LabelEncoder()
label_encoder_crop_type = LabelEncoder()
label_encoder_crop = LabelEncoder()

data['State_Name'] = label_encoder_state.fit_transform(data['State_Name'])
data['Crop_Type'] = label_encoder_crop_type.fit_transform(data['Crop_Type'])
data['Crop'] = label_encoder_crop.fit_transform(data['Crop'])

# Separate features and target variable
X = data[['State_Name', 'Crop_Type', 'N', 'P', 'K', 'pH', 'rainfall', 'temperature']] #separate features
y = data['Crop']  # target variables

# Step 2: Train-Test Split 
# ( #Splits the dataset into training 80% and testing 20% sets )
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the Model
model = RandomForestClassifier(random_state=42, n_estimators=100) #Specifies the number of trees in the forest.
model.fit(X_train, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=label_encoder_crop.classes_))

# Step 5: Save the Model and Encoders
joblib.dump(model, 'model/crop_recommendation_model.pkl')
joblib.dump(label_encoder_crop, 'model/label_encoder_crop.pkl')
joblib.dump(label_encoder_crop_type, 'model/label_encoder_crop_type.pkl')
joblib.dump(label_encoder_state, 'model/label_encoder_state.pkl')

print("Model and encoders saved successfully!")
