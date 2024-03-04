import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from keras.datasets import cifar10
from keras.datasets import mnist


def idea1(dataset_name, df0, i_sample, j_sample, x_test, y_test, it=100):
    df1 = df0.where(df0.iloc[:, -1] == 0).dropna().sample(i_sample)
    df2 = df0.where(df0.iloc[:, -1] == 1).dropna().sample(j_sample)
    train = pd.concat([df1, df2])

    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1].tolist()

    # weights = {"class_1": len(x_train) / (2 * i_sample), "class_2": len(x_train) / (2 * j_sample)}
    clf = LogisticRegression(class_weight="balanced", max_iter=it)
    clf.fit(x_train, y_train)

    print(f'Accuracy in {dataset_name} with {i_sample} samples from "class_1", {j_sample} samples from "class_2" =', clf.score(x_test, y_test))


def idea2(dataset_name, df0, i_sample, j_sample, x_test, y_test, it=100):
    df1 = df0.where(df0.iloc[:, -1] == 0).dropna().sample(i_sample)
    df1 = pd.concat([df1]*(j_sample//i_sample), ignore_index=True)
    df2 = df0.where(df0.iloc[:, -1] == 1).dropna().sample(j_sample)
    train = pd.concat([df1, df2])

    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1].tolist()

    clf = LogisticRegression(max_iter=it)
    clf.fit(x_train, y_train)

    print(f'Accuracy in {dataset_name} with {i_sample} samples from "class_1", {j_sample} samples from "class_2" =', clf.score(x_test, y_test))


def idea3(dataset_name, df0, i_sample, j_sample, x_test, y_test, it=100):
    df1 = df0.where(df0.iloc[:, -1] == 0).dropna().sample(i_sample)
    df2 = df0.where(df0.iloc[:, -1] == 1).dropna().sample(j_sample)
    train = pd.concat([df1, df2])

    x_train = train.iloc[:, :-1]
    y_train = train.iloc[:, -1].tolist()

    clf = LogisticRegression(penalty='l1', solver='liblinear', class_weight='balanced', max_iter=it)
    clf.fit(x_train, y_train)

    print(f'Accuracy in {dataset_name} with {i_sample} samples from "class_1", {j_sample} samples from "class_2" =', clf.score(x_test, y_test))


def run():
    # Load the Iris dataset
    iris_df = pd.read_csv("iris.csv")
    iris_df = iris_df.drop("Id", axis=1)

    for i in iris_df.index:
        if iris_df.at[i, 'Species'] == 'Iris-setosa':
            iris_df.at[i, 'Species'] = 0
        else:
            iris_df.at[i, 'Species'] = 1


    # idea 1: with different Imbalance samples
    print("idea 1 in Iris Dataset:")
    # a
    idea1('Iris Dataset', iris_df, 1, 99, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # b
    idea1('Iris Dataset', iris_df, 5, 95, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # c
    idea1('Iris Dataset', iris_df, 10, 90, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # d
    idea1('Iris Dataset', iris_df, 20, 80, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # e
    idea1('Iris Dataset', iris_df, 30, 70, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())

    # idea 2: with different Imbalance samples
    print("idea 2 in Iris Dataset:")
    # a
    idea2('Iris Dataset', iris_df, 1, 99, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # b
    idea2('Iris Dataset', iris_df, 5, 95, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # c
    idea2('Iris Dataset', iris_df, 10, 90, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # d
    idea2('Iris Dataset', iris_df, 20, 80, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # e
    idea2('Iris Dataset', iris_df, 30, 70, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())

    # idea 3: with different Imbalance samples
    print("idea 3 in Iris Dataset:")
    # a
    idea3('Iris Dataset', iris_df, 1, 99, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # b
    idea3('Iris Dataset', iris_df, 5, 95, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # c
    idea3('Iris Dataset', iris_df, 10, 90, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # d
    idea3('Iris Dataset', iris_df, 20, 80, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())
    # e
    idea3('Iris Dataset', iris_df, 30, 70, iris_df.iloc[:, :-1], iris_df.iloc[:, -1].tolist())

    # ------------------------------------------------------------------------------------------------------------------

    # Load the Mnist dataset
    (mnist_x_train, mnist_y_train), (mnist_x_test, mnist_y_test) = mnist.load_data()

    # Flatten the images to 1D arrays
    mnist_x_train = mnist_x_train.reshape(mnist_x_train.shape[0], -1)

    # Convert labels to 1D arrays
    mnist_y_train = mnist_y_train.ravel()

    for i in range(len(mnist_y_train)):
        if mnist_y_train[i] in range(5):
            mnist_y_train[i] = 0
        else:
            mnist_y_train[i] = 1

    # Normalize the data
    mnist_x_train = mnist_x_train.astype('float32') / 255.0

    mnist_df_x_train = pd.DataFrame(mnist_x_train)
    mnist_df_y_train = pd.DataFrame(mnist_y_train)
    mnist_df = pd.concat([mnist_df_x_train, mnist_df_y_train], axis=1)

    # idea 1: with different Imbalance samples
    print("idea 1 Mnist Dataset:")
    # a
    idea1('Mnist Dataset', mnist_df, 200, 19800, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # b
    idea1('Mnist Dataset', mnist_df, 800, 19200, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # c
    idea1('Mnist Dataset', mnist_df, 2000, 18000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # d
    idea1('Mnist Dataset', mnist_df, 4000, 16000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # e
    idea1('Mnist Dataset', mnist_df, 6000, 12000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)

    # idea 2: with different Imbalance samples
    print("idea 2 Mnist Dataset:")
    # a
    idea2('Mnist Dataset', mnist_df, 200, 19800, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # b
    idea2('Mnist Dataset', mnist_df, 800, 19200, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # c
    idea2('Mnist Dataset', mnist_df, 2000, 18000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # d
    idea2('Mnist Dataset', mnist_df, 4000, 16000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # e
    idea2('Mnist Dataset', mnist_df, 6000, 12000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)

    # idea 3: with different Imbalance samples
    print("idea 3 Mnist Dataset:")
    # a
    idea3('Mnist Dataset', mnist_df, 200, 19800, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # b
    idea3('Mnist Dataset', mnist_df, 800, 19200, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # c
    idea3('Mnist Dataset', mnist_df, 2000, 18000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # d
    idea3('Mnist Dataset', mnist_df, 4000, 16000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)
    # e
    idea3('Mnist Dataset', mnist_df, 6000, 12000, mnist_df.iloc[:, :-1], mnist_df.iloc[:, -1].tolist(), it=1000)

    # ------------------------------------------------------------------------------------------------------------------

    # Load the CIFAR-10 dataset
    (cifar10_x_train, cifar10_y_train), (cifar10_x_test, cifar10_y_test) = cifar10.load_data()

    # Flatten the images to 1D arrays
    cifar10_x_train = cifar10_x_train.reshape(cifar10_x_train.shape[0], -1)

    # Convert labels to 1D arrays
    cifar10_y_train = cifar10_y_train.ravel()

    for i in range(len(cifar10_y_train)):
        if cifar10_y_train[i] in range(5):
            cifar10_y_train[i] = 0
        else:
            cifar10_y_train[i] = 1

    # Normalize the data
    cifar10_x_train = cifar10_x_train.astype('float32') / 255.0

    cifar10_df_x_train = pd.DataFrame(cifar10_x_train)
    cifar10_df_y_train = pd.DataFrame(cifar10_y_train)
    cifar10_df = pd.concat([cifar10_df_x_train, cifar10_df_y_train], axis=1)

    # idea 1: with different Imbalance samples
    print("idea 1 Cifar10 Dataset:")
    # a
    idea1('Cifar10 Dataset', cifar10_df, 200, 19800, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # b
    idea1('Cifar10 Dataset', cifar10_df, 800, 19200, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # c
    idea1('Cifar10 Dataset', cifar10_df, 2000, 18000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # d
    idea1('Cifar10 Dataset', cifar10_df, 4000, 16000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # e
    idea1('Cifar10 Dataset', cifar10_df, 6000, 12000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)

    # idea 2: with different Imbalance samples
    print("idea 2 Cifar10 Dataset:")
    # a
    idea2('Cifar10 Dataset', cifar10_df, 200, 19800, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # b
    idea2('Cifar10 Dataset', cifar10_df, 800, 19200, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # c
    idea2('Cifar10 Dataset', cifar10_df, 2000, 18000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # d
    idea2('Cifar10 Dataset', cifar10_df, 4000, 16000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # e
    idea2('Cifar10 Dataset', cifar10_df, 6000, 12000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)

    # idea 3: with different Imbalance samples
    print("idea 3 Cifar10 Dataset:")
    # a
    idea3('Cifar10 Dataset', cifar10_df, 200, 19800, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # b
    idea3('Cifar10 Dataset', cifar10_df, 800, 19200, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # c
    idea3('Cifar10 Dataset', cifar10_df, 2000, 18000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # d
    idea3('Cifar10 Dataset', cifar10_df, 4000, 16000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)
    # e
    idea3('Cifar10 Dataset', cifar10_df, 6000, 12000, cifar10_df.iloc[:, :-1], cifar10_df.iloc[:, -1].tolist(), it=10000)


if __name__ == '__main__':
    run()
