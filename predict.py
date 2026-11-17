import pandas as pd
import pickle
with open('models/student_model.pkl', 'rb') as file:
    model = pickle.load(file)
new_data = pd.read_csv('Dataset/new_student.csv')
X_new = new_data[
    ['study_hours', 'attendance', 'previous_marks']
]
predictions = model.predict(X_new)
new_data['predicted_result'] = predictions
print("Predicted Student Results:")
print(new_data)
new_data.to_csv(
    'Dataset/student_predictions.csv',
    index=False
)
print("\nPredictions saved successfully!")