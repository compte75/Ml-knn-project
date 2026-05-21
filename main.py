from data_loader import load_data
from Knn import KNN
from sklearn.model_selection import train_test_split

X, Y = load_data("bienetre.xlsx")
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

knn = KNN(K=5)
knn.fit(X_train, Y_train)
print(knn.evaluate(X_test, Y_test))