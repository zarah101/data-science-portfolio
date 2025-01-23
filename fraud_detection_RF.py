# Fraud Detection Model
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Classifier Libraries
from sklearn.ensemble import RandomForestClassifier
import collections


# Other Libraries
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# Load the data
df = pd.read_csv('creditcard.csv')
print(df.head())

print(df.describe())

# Let's check for null values
print(df.isnull().sum().max())

# Let's see the distribution of the data
count_classes = pd.value_counts(df['Class'], sort = True).sort_index()
count_classes.plot(kind = 'bar')
plt.title("Fraud Class Distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
plt.show()

# Our data is heavily skewed, let's look at the numbers
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')



# Let's see the distribution of Transaction Time and Amounts
fig, ax = plt.subplots(1, 2, figsize=(18,4))

amount_val = df['Amount'].values
time_val = df['Time'].values

sns.distplot(amount_val, ax=ax[0], color='r')
ax[0].set_title('Distribution of Transaction Amount', fontsize=14)
ax[0].set_xlim([min(amount_val), max(amount_val)])

sns.distplot(time_val, ax=ax[1], color='b')
ax[1].set_title('Distribution of Transaction Time', fontsize=14)
ax[1].set_xlim([min(time_val), max(time_val)])

plt.show()


# We see that 99.83% of the data is not fraud, which is good! But we need to balance the data
# Let's shuffle the data before creating the subsamples
df = df.sample(frac=1)

# amount of fraud classes 492 rows.
fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:492]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])

# Shuffle dataframe rows
new_df = normal_distributed_df.sample(frac=1, random_state=42)

print(new_df.head())

# Let's see the distribution of the data
print('Distribution of the Classes in the subsample dataset')
print(new_df['Class'].value_counts()/len(new_df))

sns.countplot(x='Class', data=new_df)
plt.title('Equally Distributed Classes', fontsize=14)
plt.show()


# Now let's see the correlation between the variables
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(24,20))

# Entire DataFrame Correlation
corr = df.corr()
sns.heatmap(corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax1)
ax1.set_title("Imbalanced Correlation Matrix", fontsize=14)

#Sub Sample Correlation Matrix
sub_sample_corr = new_df.corr()
sns.heatmap(sub_sample_corr, cmap='coolwarm_r', annot_kws={'size':20}, ax=ax2)
ax2.set_title('SubSample Correlation Matrix', fontsize=14)
plt.show()


# Now we want to train and test the data
X = new_df.drop('Class', axis=1)
y = new_df['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Let's see the shape of the data
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Let's start modeling
# Since our dataset was imbalanced, we will focus on Random Forest
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(class_weight='balanced', random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Let's see the results
# Let's see the accuracy score
print('Accuracy Score: ', accuracy_score(y_test, y_pred))

# Let's see the precision score
print('Precision Score: ', precision_score(y_test, y_pred))

# Let's see the recall score
print('Recall Score: ', recall_score(y_test, y_pred))

# Let's see the f1 score
print('F1 Score: ', f1_score(y_test, y_pred))

# Let's see the classification report
print('Classification Report: ', classification_report(y_test, y_pred))

# Let's see the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Let's see the feature importance
feature_importances = rf.feature_importances_
features = X_train.columns
data = {'features': features, 'importance': feature_importances}
df = pd.DataFrame(data)
df = df.sort_values(by='importance', ascending=False)
print(df)

# Let's visualize the feature importance
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='features', data=df)
plt.title('Feature Importance')
plt.show()






