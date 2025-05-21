import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load data
df = pd.read_csv('placement_data.csv')

# Step 2: Preprocessing

# Encode categorical variables (degree, branch, skills)
le_degree = LabelEncoder()
df['degree_enc'] = le_degree.fit_transform(df['degree'])

le_branch = LabelEncoder()
df['branch_enc'] = le_branch.fit_transform(df['branch'])

# For skills: convert list of skills separated by ';' into multiple binary columns (one-hot)
skills_set = set()
for skills in df['skills']:
    for skill in skills.split(';'):
        skills_set.add(skill)

for skill in skills_set:
    df[f'skill_{skill}'] = df['skills'].apply(lambda x: 1 if skill in x.split(';') else 0)

# Features to use
feature_cols = ['degree_enc', 'branch_enc', 'percentage', 'backlogs', 'internships'] + [f'skill_{skill}' for skill in skills_set]

X = df[feature_cols]
y = df['placed']

# Step 3: Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scaling numeric features
scaler = StandardScaler()
num_cols = ['percentage', 'backlogs', 'internships']
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# Step 5: Train Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 6: Predict and evaluate
y_pred = clf.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Placed', 'Placed'], yticklabels=['Not Placed', 'Placed'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
