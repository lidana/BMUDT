import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot(X, y, dataname):
    tsne = TSNE(n_components=2)
    x_tsne = tsne.fit_transform(X)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y)
    plt.legend(["class 0", "class 1", "mixup"])
    #plt.show()
    f = plt.gcf()
    f.savefig('figure/{}.png'.format(dataname))
    print('figure/{}.png saved!'.format(dataname))
    f.clear()


def plot_mixup(X, y, mix_x, mix_y, dataname):
    tsne = TSNE(n_components=2, method='exact', init='pca', random_state=0, perplexity=5)
    x_tsne = tsne.fit_transform(X)
    plt.scatter(x_tsne[:, 0], x_tsne[:, 1], c=y, label="samples")
    mix_tsne = tsne.fit_transform(mix_x)
    plt.scatter(mix_tsne[:, 0], mix_tsne[:, 1], c=mix_y, label="mixup")
    #plt.show()
    f = plt.gcf()
    f.savefig('figure_mixup/{}.png'.format(dataname))
    print('figure_mixup/{}.png saved!'.format(dataname))
    f.clear()


def generate_data(data):
    X1 = data[data[:, -1] == 0, :]
    X2 = data[data[:, -1] == 1, :]
    mix_num = min(X1.shape[0], X2.shape[0])
    mix = (X1[0:mix_num, :] + X2[0:mix_num, :]) / 2
    mix_x = np.delete(mix, [-1], axis=1)
    mix_y = np.full(mix_num, 2)
    y = data[:, -1]
    X = np.delete(data, [-1], axis=1)
    X = np.concatenate((X, mix_x), axis=0)
    y = np.concatenate((y, mix_y), axis=0)
    return X, y


def generate_data_mixup(data):
    X1 = data[data[:, -1] == 0, :]
    X2 = data[data[:, -1] == 1, :]
    mix_num = min(X1.shape[0], X2.shape[0])
    mix = (X1[0:mix_num, :] + X2[0:mix_num, :]) / 2
    mix_x = np.delete(mix, [-1], axis=1)
    mix_y = np.full(mix_num, 2)
    y = data[:, -1]
    X = np.delete(data, [-1], axis=1)
    return X, y, mix_x, mix_y


if __name__ == '__main__':
    dataset = ['wilt', 'KungChi3', 'KnuggetChase3', 'cloud', 'SPECTF', 'prnn_crabs', 'haberman', 'stock',
               'qualitative-bankruptcy', 'xd6', 'tokyo1', 'kc1-top5', 'diabetes', 'datatrieve', 'planning-relax',
               'banknote-authentication', 'ilpd', 'sonar', 'spambase', 'heart-statlog', 'monks-problems-1',
               'monks-problems-2', 'monks-problems-3', 'mfeat-morphological', 'ionosphere',
               'blood-transfusion-service-center', 'balloons', 'breast-cancer-wisc-diag_R', 'breast-cancer-wisc-prog_R',
               'conn-bench-sonar-mines-rocks_R', 'cylinder-bands_R', 'echocardiogram_R', 'fertility_R',
               'haberman-survival_R', 'heart-hungarian_R', 'hepatitis_R', 'ilpd-indian-liver_R', 'ionosphere_R',
               'mammographic_R', 'molec-biol-promoter_R', 'oocytes_merluccius_nucleus_4d_R',
               'oocytes_trisopterus_nucleus_2f_R', 'parkinsons_R', 'pima_R', 'pittsburg-bridges-T-OR-D_R', 'planning_R',
               'post-operative_R', 'statlog-australian-credit_R', 'statlog-german-credit_R', 'statlog-heart_R',
               'vertebral-column-2clases_R']
    for idx in range(51):
        if idx < 26:
            datapath = "DATAOPENML/" + dataset[idx] + ".csv"
        else:
            datapath = "DATASET/" + dataset[idx] + ".dat"
            df = pd.read_csv(datapath)
            data = df.to_numpy()
            X, y = generate_data(data)
            plot(X, y, dataset[idx])
            #X, y, mix_x, mix_y = generate_data_mixup(data)
            #plot_mixup(X, y, mix_x, mix_y, dataset[idx])
    print("DONE!")

