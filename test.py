from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd

# Sample data
data = pd.DataFrame({
    'x1': [1, 2, 3, 4, 5],
    'x2': [1, 3, 2, 4, 5],
    'x3': [2, 3, 4, 5, 6],
    'y': ['A', 'B', 'A', 'B', 'A']
})

# Separate features and target variable
X = data[['x1', 'x2', 'x3']]
y = data['y']

# Create and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Get the linear discriminants (similar to loadings in PCA)
linear_discriminants = lda.transform(X)

# Display the linear discriminants
print(linear_discriminants)
