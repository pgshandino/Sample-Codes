# Load the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_df = pd.read_csv(url)
# Separate features and target variable
X = titanic_df.drop('Survived', axis=1)
y = titanic_df['Survived']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns
# Create a ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),  # Impute missing numerical values
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ])
# Create a pipeline with preprocessing and modeling steps
pipeline = Pipeline([
    ('preprocessor', preprocessor),                # Apply preprocessing
    ('classifier', RandomForestClassifier())       # Apply a random forest classifier
])
# Fit the pipeline on the training data
pipeline.fit(X_train, y_train)
# Make predictions on the testing data
predictions = pipeline.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy}')
