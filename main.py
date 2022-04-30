import os
import numpy
import sys
import zlib
import cv2
import time
import random
from threading import Thread
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
try:
    import cPickle as pickle
    from urllib2 import urlopen
    from Queue import Queue
except ImportError:
    import pickle
    from urllib.request import urlopen
    from queue import Queue

CLUSTER_SEED = 24
CLUSTERS_NUMBER = 100
BAYES_ALPHA = 0.1
ADA_BOOST_ESTIMATORS = 110


def descriptors(img):
    # меняем размер файла, тк на файлах с большим разрешением дескрипторы считаются дольше
    if img.shape[1] > 1000:
        cf = 1000.0 / img.shape[1]
        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]), img.shape[2])
        img.resize(newSize, refcheck=False)
    orb = cv2.ORB_create()  #  создание orb объекта
    kp, des = orb.detectAndCompute(img, None)  #  нахождение ключевых точек и дескрипторов
    hist = getColorHist(img)
    # img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    # plt.imshow(img2, cmap='gray'), plt.title('ORB'), plt.axis('off')
    # plt.show()
    return des, hist


def getColorHist(img): #  считаем цветовую гистограмму для второго классификатора
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    return dist


def addDescriptors(totalDescriptors, sample):
    for descriptor in sample[0]:
        totalDescriptors.append(descriptor)


def calculteCounts(samples, counts, counts1, clusters, currentDescr, currentSample):
    for s in samples:
        currentCounts = {}
        for _ in s[0]:
            currentCounts[clusters[currentDescr]] = currentCounts.get(clusters[currentDescr], 0) + 1
            currentDescr += 1
        for clu, cnt in currentCounts.items():
            counts[currentSample, clu] = cnt
        for i, histCnt in enumerate(s[1]):
            counts1[currentSample, i] = histCnt[0]
        currentSample += 1
    return currentDescr, currentSample


def LoadSamples(img):
    img = cv2.imread(directory + image)
    des, hist = descriptors(img)
    return des, hist

# загрузка обучающей выборки и нахождение дескрипторов
# положительные и отрицательные примеры
directory = 'images/triangle/'
totalDescriptors = []

samples = {}
directory = 'images/'
for image in os.listdir(directory):
    s = LoadSamples(image)
    name = os.path.basename(directory + image)
    name = name[:name.index('#')]
    if samples.get(name, 0) != 0:
        samples[name].append(s)
    else:
        samples[name] = [s]
    addDescriptors(totalDescriptors, s)
num = 0
for key, value in samples.items():
    print(f'Название фигуры: {key}, количество изображений с ней: {len(value)}')
    num += len(value)
#  создание всех необходимых классификаторов
#  kmeans — для кластеризации
#  TfidfTransformer, MultinomialBN, AdaBoostClassifier — для классификации
kmeans = MiniBatchKMeans(n_clusters=CLUSTERS_NUMBER, random_state=CLUSTER_SEED, verbose=True)
_tfidf = TfidfTransformer()
_tfidf1 = TfidfTransformer()
clf = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
clf1 = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
#  обучение модели
kmeans.fit(totalDescriptors)
clusters = kmeans.predict(totalDescriptors) #  список кластеров с соответсвием дескрипторам
clusters = numpy.array(clusters)
totalSamplesNumber = num
#  создание двух матрицы, подсчет частоты встречаемости
counts = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER)) #  сколько раз встретились дескрипторы из каждого класстера для каждой картинки
counts1 = lil_matrix((totalSamplesNumber, 256)) #  сколько раз встретился каждый цвет для каждой картинки
currentDescr = 0
currentSample = 0
for i in samples.values():
    currentDescr, currentSample = calculteCounts(i, counts, counts1, clusters, currentDescr, currentSample)
counts = csr_matrix(counts)
counts1 = csr_matrix(counts1)
tfidf = _tfidf.fit_transform(counts)  #  преобразование матриц
tfidf1 = _tfidf1.fit_transform(counts1)
classes = []
for key, value in samples.items():
    classes += [key] * len(value)
clf.fit(tfidf, classes) #  обучение байесовского классификатора
clf1.fit(tfidf1, classes)


# тестовая выборка для предсказаний
# действия аналогичные обучению модели
directory = 'testing/test/'
samples1 = []
totalDescriptors = []
for image in os.listdir(directory):
    s = LoadSamples(image)
    samples1.append(s)
    addDescriptors(totalDescriptors, s)
clusters = kmeans.predict(totalDescriptors)
counts = lil_matrix((len(samples1), CLUSTERS_NUMBER))
counts1 = lil_matrix((len(samples1), 256))
currentDescr = 0
currentSample = 0
currentDescr, currentSample = calculteCounts(samples1, counts, counts1, clusters, currentDescr, currentSample)
counts = csr_matrix(counts)
counts1 = csr_matrix(counts1)

tfidf = _tfidf.transform(counts)
tfidf1 = _tfidf1.transform(counts1)

# предсказание результата по классификатору с дескрипторами
weights = clf.predict_log_proba(tfidf)
weights1 = clf1.predict_log_proba(tfidf1)
predictions = []
figures = [i for i in samples.keys()]
for i in weights:
    f = 0
    for j in range(len(i)):
        if i[j] == max(i):
            f = j
            break
        f += 1
    predictions.append(figures[f])
print(predictions)
