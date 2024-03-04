import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.datasets import cifar10
from keras.datasets import mnist


def idea1(dataset_name, x_train, y_train, x_test, y_test, it=100):
    clf = LogisticRegression(class_weight="balanced", max_iter=it)
    clf.fit(x_train, y_train)

    print(f'Accuracy in {dataset_name} =', clf.score(x_test, y_test))


def idea2(dataset_name, x_train, y_train, x_test, y_test, it=100):
    clf = LogisticRegression(max_iter=it)
    clf.fit(x_train, y_train)

    print(f'Accuracy in {dataset_name} =', clf.score(x_test, y_test))


def idea3(dataset_name, x_train, y_train, x_test, y_test, it=100):
    clf = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=it)
    clf.fit(x_train, y_train)

    print(f'Accuracy in {dataset_name} =', clf.score(x_test, y_test))


def run():
    # Load the Iris dataset
    iris_df = pd.read_csv("iris.csv")
    iris_df = iris_df.drop("Id", axis=1)

    df1 = iris_df.where(iris_df.iloc[:, -1] == 'Iris-setosa').dropna().sample(20)
    df2 = iris_df.where(iris_df.iloc[:, -1] == 'Iris-versicolor').dropna().sample(30)
    df3 = iris_df.where(iris_df.iloc[:, -1] == 'Iris-virginica').dropna().sample(50)
    train = pd.concat([df1, df2, df3])

    # idea 1:
    print("idea 1 in Iris Dataset:")
    idea1('Iris Dataset', train.iloc[:, :-1], train.iloc[:, -1], iris_df.iloc[:, :-1], iris_df.iloc[:, -1])

    # idea 2:
    print("idea 2 in Iris Dataset:")
    idea2_df1 = pd.concat([df1] * (5 // 2), ignore_index=True)
    idea2_df2 = pd.concat([df2] * (5 // 3), ignore_index=True)
    idea2_train = pd.concat([idea2_df1, idea2_df2, df3])
    idea2('Iris Dataset', idea2_train.iloc[:, :-1], idea2_train.iloc[:, -1], iris_df.iloc[:, :-1], iris_df.iloc[:, -1], it=1000000)

    # idea 3:
    print("idea 3 in Iris Dataset:")
    idea3('Iris Dataset', train.iloc[:, :-1], train.iloc[:, -1], iris_df.iloc[:, :-1], iris_df.iloc[:, -1], it=100000000)

    # ------------------------------------------------------------------------------------------------------------------

    # Load the Mnist dataset
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

    # Flatten the images to 1D arrays
    mnist_x_train = mnist_x_train.reshape(mnist_x_train.shape[0], -1)

    # Convert labels to 1D arrays
    mnist_y_train = mnist_y_train.ravel()

    # Normalize the data
    mnist_x_train = mnist_x_train.astype('float32') / 255.0

    mnist_df_x_train = pd.DataFrame(mnist_x_train)
    mnist_df_y_train = pd.DataFrame(mnist_y_train)
    mnist_df = pd.concat([mnist_df_x_train, mnist_df_y_train], axis=1)

    df1 = mnist_df.where(mnist_df.iloc[:, -1] == 0).dropna().sample(500)
    df2 = mnist_df.where(mnist_df.iloc[:, -1] == 1).dropna().sample(1000)
    df3 = mnist_df.where(mnist_df.iloc[:, -1] == 2).dropna().sample(1500)
    df4 = mnist_df.where(mnist_df.iloc[:, -1] == 3).dropna().sample(2000)
    df5 = mnist_df.where(mnist_df.iloc[:, -1] == 4).dropna().sample(2500)
    df6 = mnist_df.where(mnist_df.iloc[:, -1] == 5).dropna().sample(3000)
    df7 = mnist_df.where(mnist_df.iloc[:, -1] == 6).dropna().sample(3500)
    df8 = mnist_df.where(mnist_df.iloc[:, -1] == 7).dropna().sample(4000)
    df9 = mnist_df.where(mnist_df.iloc[:, -1] == 8).dropna().sample(4500)
    df10 = mnist_df.where(mnist_df.iloc[:, -1] == 9).dropna().sample(5000)

    train = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10])

    # idea 1:
    print("idea 1 in Mnist Dataset:")
    idea1('Mnist Dataset', train.iloc[:, :-1], train.iloc[:, -1], mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1], it=1000)
    # idea 3:
    print("idea 3 in Mnist Dataset:")
    idea1('Mnist Dataset', train.iloc[:, :-1], train.iloc[:, -1], mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1], it=1000)

    # ------------------------------------------------------------------------------------------------------------------

    # Load the CIFAR-10 dataset
    (cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()

    # Flatten the images to 1D arrays
    cifar10_x_train = cifar10_x_train.reshape(cifar10_x_train.shape[0], -1)

    # Convert labels to 1D arrays
    cifar10_y_train = cifar10_y_train.ravel()

    # Normalize the data
    cifar10_x_train = cifar10_x_train.astype('float32') / 255.0

    cifar10_df_x_train = pd.DataFrame(cifar10_x_train)
    cifar10_df_y_train = pd.DataFrame(cifar10_y_train)
    cifar10_df = pd.concat([cifar10_df_x_train, cifar10_df_y_train], axis=1)

    df1 = cifar10_df.where(cifar10_df.iloc[:, -1] == 0).dropna().sample(500)
    df2 = cifar10_df.where(cifar10_df.iloc[:, -1] == 1).dropna().sample(1000)
    df3 = cifar10_df.where(cifar10_df.iloc[:, -1] == 2).dropna().sample(1500)
    df4 = cifar10_df.where(cifar10_df.iloc[:, -1] == 3).dropna().sample(2000)
    df5 = cifar10_df.where(cifar10_df.iloc[:, -1] == 4).dropna().sample(2500)
    df6 = cifar10_df.where(cifar10_df.iloc[:, -1] == 5).dropna().sample(3000)
    df7 = cifar10_df.where(cifar10_df.iloc[:, -1] == 6).dropna().sample(3500)
    df8 = cifar10_df.where(cifar10_df.iloc[:, -1] == 7).dropna().sample(4000)
    df9 = cifar10_df.where(cifar10_df.iloc[:, -1] == 8).dropna().sample(4500)
    df10 = cifar10_df.where(cifar10_df.iloc[:, -1] == 9).dropna().sample(5000)

    train = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8, df9, df10])

    # idea 1:
    print("idea 1 in Cifar10 Dataset:")
    idea1('Cifar10 Dataset', train.iloc[:, :-1], train.iloc[:, -1], cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1], it=100000)
    # idea 3:
    print("idea 3 in Cifar10 Dataset:")
    idea1('Cifar10 Dataset', train.iloc[:, :-1], train.iloc[:, -1], cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1], it=100000)


if __name__ == '__main__':
    run()
