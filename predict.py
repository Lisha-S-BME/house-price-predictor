
# predict.py
# House Price Prediction Script
# Usage: python predict.py

import pickle
import pandas as pd
import numpy as np
import os

print("Loading model...")

# Load the saved model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

print("Model loaded successfully")

# Check if input file exists
input_file = 'new_properties.csv'

if not os.path.exists(input_file):
    print(f"Error: {input_file} not found")
    print("Creating sample input file for testing...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        'area': [5000, 7500, 10000],
        'bedrooms': [3, 4, 5],
        'bathrooms': [2, 3, 4],
        'stories': [2, 2, 3],
        'parking': [1, 2, 2],
        'mainroad_yes': [1, 1, 1],
        'guestroom_yes': [0, 1, 0],
        'basement_yes': [0, 0, 1],
        'hotwaterheating_yes': [0, 0, 0],
        'airconditioning_yes': [1, 1, 1],
        'prefarea_yes': [1, 0, 1],
        'furnishingstatus_semi-furnished': [0, 1, 0],
        'furnishingstatus_unfurnished': [0, 0, 1],
        'area_per_room': [5000/4, 7500/5, 10000/6]
    })
    sample_data.to_csv(input_file, index=False)
    print(f"Created {input_file} with {len(sample_data)} sample properties")

# Load new data
print(f"\nLoading data from {input_file}...")
new_data = pd.read_csv(input_file)
print(f"Loaded {len(new_data)} properties")

# Make predictions
print("Making predictions...")
predictions = model.predict(new_data)

# Add predictions to dataframe
new_data['predicted_price'] = predictions

# Save results
output_file = 'predictions_output.csv'
new_data.to_csv(output_file, index=False)

print(f"\nPredictions saved to {output_file}")
print("\nResults:")
print("-" * 60)
for i, row in new_data.iterrows():
    print(f"Property {i+1}: Area={row['area']}, Bedrooms={row['bedrooms']}")
    print(f"  Predicted Price: ${row['predicted_price']:,.2f}")
    print("-" * 60)
