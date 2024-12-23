import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

iris_data = load_iris()

df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
df['species'] = iris_data.target_names[iris_data.target]

st.write("### Iris Dataset Overview")
st.write(df.head())

st.write("### Dataset Statistics")
st.write(df.describe())

st.write("### Missing Values in Dataset")
st.write(df.isnull().sum())

st.write("### Data Visualizations")

st.write("#### Pair Plot of Features")
sns.pairplot(df, hue='species', palette='Set1')
plt.suptitle('Pairwise Relationships between Features', y=1.02)
st.pyplot(plt)

st.write("#### Correlation Heatmap")
correlation_matrix = df.drop('species', axis=1).corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap of Features')
st.pyplot(plt)

st.write("#### Box Plots of Features")
plt.figure(figsize=(12, 8))
features = df.columns[:-1]
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.boxplot(x='species', y=feature, data=df, palette='Set1')
    plt.title(f'Box Plot of {feature}')
plt.tight_layout()
st.pyplot(plt)

st.write("#### Violin Plots of Features")
plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    plt.subplot(2, 2, i+1)
    sns.violinplot(x='species', y=feature, data=df, palette='Set1')
    plt.title(f'Violin Plot of {feature}')
plt.tight_layout()
st.pyplot(plt)

st.write("#### Histograms of Features")
df.drop('species', axis=1).hist(bins=20, figsize=(12, 8), color='lightblue', edgecolor='black')
plt.suptitle('Histograms of Features')
plt.tight_layout()
st.pyplot(plt)

st.write("#### PairGrid of Features")
sns.PairGrid(df, hue='species', palette='Set1').map_lower(sns.kdeplot, cmap='Blues_d').map_diag(sns.histplot)
plt.suptitle('Pairwise Relationships with KDE on the Lower Triangle', y=1.02)
st.pyplot(plt)

st.write("### PCA for Dimensionality Reduction")
pca = PCA(n_components=2)
principal_components = pca.fit_transform(df.drop('species', axis=1))

pca_df = pd.DataFrame(principal_components, columns=['PC1', 'PC2'])
pca_df['species'] = df['species']

plt.figure(figsize=(8, 6))
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df, palette='Set1', s=100, alpha=0.7)
plt.title('PCA of Iris Dataset')
st.pyplot(plt)

st.write("### Standardizing the Data (Setting Mean to 0 and Standard Deviation to 1)")
scaler = StandardScaler()
df_scaled = df.drop('species', axis=1)
df_scaled = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df.columns[:-1])

df_scaled['species'] = df['species']
st.write(df_scaled.head())

X = df_scaled.drop('species', axis=1)
y = df_scaled['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

model = DecisionTreeClassifier(random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"### Model Accuracy: {accuracy:.2f}")
st.write("### Classification Report")
st.write(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris_data.target_names, yticklabels=iris_data.target_names)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
st.pyplot(plt)

new_data = [[5.1, 3.5, 1.4, 0.2]]
new_data_scaled = scaler.transform(new_data)
prediction = model.predict(new_data_scaled)
st.write(f"Predicted species for new data: {prediction[0]}")

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=0),
    'Random Forest': RandomForestClassifier(random_state=0),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machine': SVC(random_state=0)
}

model_accuracies = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy

plt.figure(figsize=(10, 6))
sns.barplot(x=list(model_accuracies.keys()), y=list(model_accuracies.values()))
plt.title('Model Accuracy Comparison')
plt.xlabel('Model')
plt.ylabel('Accuracy')
st.pyplot(plt)

st.write("### Iris Flower Prediction App")

st.write("""
This app predicts the species of Iris flowers based on the following features:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Please enter the values for each feature and press enter to get the predicted species.
""")

sepal_length = st.text_input("Enter Sepal Length (cm):", "5.1")
sepal_width = st.text_input("Enter Sepal Width (cm):", "3.5")
petal_length = st.text_input("Enter Petal Length (cm):", "1.4")
petal_width = st.text_input("Enter Petal Width (cm):", "0.2")
sepal_length = float(sepal_length)
sepal_width = float(sepal_width)
petal_length = float(petal_length)
petal_width = float(petal_width)

if all([sepal_length, sepal_width, petal_length, petal_width]):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    scaled_input = scaler.transform(input_data)
    prediction = model.predict(scaled_input)

    species = prediction[0]

    st.write(f"### Predicted Species: {species}")

