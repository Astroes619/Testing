import csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Load the data from the CSV file
with open('eye_tracking_data.csv', mode='r') as csv_file:
    csv_reader = csv.reader(csv_file)
    next(csv_reader) # Skip the header row
    data = []
    labels = []
    for row in csv_reader:
        data.append([float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[4])])
        labels.append(row[5])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2)

# Create a decision tree classifier and fit it to the training data
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Test the classifier on the testing data and print the accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
