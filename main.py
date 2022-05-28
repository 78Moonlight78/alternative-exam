import os
import numpy
import cv2
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

try:
    import cPickle as pickle
    from urllib2 import urlopen
    from Queue import Queue
except ImportError:
    import pickle
    from urllib.request import urlopen
    from queue import Queue

# переменные, использующиеся для k-means
CLUSTER_SEED = 50
CLUSTERS_NUMBER = 110
# переменные, использующаяся при бустинге
BAYES_ALPHA = 0.1
ADA_BOOST_ESTIMATORS = 200
# более подробно значение каждой переменной указано при ее использовании

'''
---------------------------------------------------------------------------------------------------------------
descriptors - Функция нахождения дескрипторов и гистограммы для классификатора.
---------------------------------------------------------------------------------------------------------------
Принимает: img - изображение в формате numpy-массива.
---------------------------------------------------------------------------------------------------------------
Возвращает: des_orb - массив дескрипторов, найденных с помощью ORB, 
hist - массив точек гистограммы.
---------------------------------------------------------------------------------------------------------------
Параметры .ORB_create():
nfeatures – максимальное число ключевых точек
scaleFactor – множитель для пирамиды изображений, больше единицы. Значение 2 реализует классическую пирамиду.
nlevels – число уровней в пирамиде изображений.
edgeThreshold – число пикселов у границы изображения, где ключевые точки не детектируются.
firstLevel – оставить нулём.
WTA_K – число точек, которое требуется для одного элемента дескриптора. Если равно 2, 
то сравнивается яркость двух случайно выбранных пикселов.
scoreType – если 0, то в качестве меры особенности используется харрис, 
иначе – мера FAST (на основе суммы модулей разностей яркостей в точках окружности).
patchSize – размер окрестности, из которой выбираются случайные пикселы для сравнения.
---------------------------------------------------------------------------------------------------------------
'''


def descriptors_and_hist(img):
    #  меняем размер файла, т.к. на файлах большого размера дескрипторы считаются дольше.
    if img.shape[1] > 1000:
        cf = 1000.0 / img.shape[1]
        new_size = (int(cf * img.shape[0]), int(cf * img.shape[1]), img.shape[2])
        img.resize(new_size, refcheck=False)
    # создание ORB-объекта.
    orb = cv2.ORB_create(nfeatures=490, scaleFactor=1.2, nlevels=4, edgeThreshold=31, firstLevel=0, WTA_K=2,
                         scoreType=0, patchSize=31)
    # нахождение ключевых точек и дескрипторов с помощью ORB.
    kp, des_orb = orb.detectAndCompute(img, None)
    # получение гистограммы
    hist = get_hist(img)
    # возвращаем кортеж найденных массивов дескрипторов.
    return des_orb, hist


'''
----------------------------------------------------------------------------------------------------------
get_hog - Функция нахождения гистограммы изображения.
----------------------------------------------------------------------------------------------------------
Принимает: img - изображение в формате numpy-массива.
----------------------------------------------------------------------------------------------------------
Возвращает: hist - массив точек гистограммы.
----------------------------------------------------------------------------------------------------------
Параметры .calcHist():
[hsv] - бинарное изображение в виде массива numpy
[0] - список каналов, используемых для вычисления гистограмм (в нашем случае канал единственный - синий).
None - дополнительная маска (8-битный массив) того же размера, что и входное изображение (отсутствует).
[256] - размеры гистограммы в каждом измерении
[0, 256] - массив границ гистограммы 
----------------------------------------------------------------------------------------------------------
'''


def get_hist(img):
    # изменяет цвета изображения под цветовую модель HSV, возвращает бинарное изображение
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    return hist


'''
---------------------------------------------------------------------------------------
addDescriptors - Функция добавления найденных на изображении пар дескриптор+гистограмма
в массив всех дескрипторов выборки.
---------------------------------------------------------------------------------------
Принимает: totalDescriptors - массив всех дескрипторов.
des_image - кортеж из дескрипторов и гистограммы конкретного изображения.
-----------------------------------------------------------------------------------------
'''


def addDescriptors(totalDescriptors, des_image):
    for descriptor in des_image[0]:
        totalDescriptors.append(descriptor)


'''
---------------------------------------------------------------------------
addNames - Функция добавления в массив наименований классов названия класса 
необходимое для соответствия массиву totalDescriptors количество раз.
---------------------------------------------------------------------------
Принимает: names - массив всех имен классов.
name - имя класса, на изображении которого были найдены дескрипторы.
des - массив найденных на изображении дескрипторов.
---------------------------------------------------------------------------
'''


def addNames(names, name, des):
    for _ in des[0]:
        names.append(name)


'''
-------------------------------------------------------------------------
calculteCounts - Функция заполнения словарей, использующихся для создания
разреженных матриц, в которых будет храниться информация 
о встречаемости кластеров. 
-------------------------------------------------------------------------
Принимает: sample - массив дескрипторов изображения
counts_orb, counts_hist, counts2 - встречаемость кластеров для каждого
алгоритма (ORB, HOG, KNN соответственно).
clusters, clusters_knn - кластеры ORB/HOG и KNN.
currentDescr - количество просмотренных дескрипторов.
currentSample - количество рассмотренных примеров.
-------------------------------------------------------------------------
'''


def calculteCounts(sample, counts_orb, counts_hist, counts_knn, kmeans_clus, knn_clus, currentDescr, currentSample):
    # просматриваем дескрипторы для каждого отдельного изображения в выборке
    for s in sample:
        # текущее количество повторений каждого дескриптора для алгоритмов k-means и knn в данном изображении
        currentCounts_km = {}
        currentCounts_knn = {}
        # рассматриваем ORB-дескрипторы данного изображения
        for _ in s[0]:
            # обновляем информацию в словарях о встречаемости дескриптора на изображении
            currentCounts_km[kmeans_clus[currentDescr]] = currentCounts_km.get(kmeans_clus[currentDescr], 0) + 1
            currentCounts_knn[knn_clus[currentDescr]] = currentCounts_knn.get(knn_clus[currentDescr], 0) + 1
            # переходим к следующему дескриптору
            currentDescr += 1
        # рассматриваем пары вида "дескриптор: кол-во его повторений"
        for clu, cnt in currentCounts_km.items():
            # обновляем информацию внутри матрицы о встречаемости каждого дескриптора-ORB в рассматриваемом изображении
            # для алгоритма k-means, а затем для алгоритма knn
            counts_orb[currentSample, clu] = cnt
        # в связи с тонкостями типов, принимаемых в дальнейшем иными методами, требуется преобразование
        keys_knn = list(currentCounts_knn.keys())
        for i in range(len(currentCounts_knn)):
            counts_knn[currentSample, i] = currentCounts_knn[keys_knn[i]]
        # заполняем матрицу гистограммы
        for i, histCnt in enumerate(s[1]):
            counts_hist[currentSample, i] = histCnt[0]
            if i == 255:
                break
        # переходим к следующему изображению
        currentSample += 1
    # возвращаем количество встреченных дескрипторов и количество пройденных примеров
    return currentDescr, currentSample


'''
-----------------------------------------------------------------------
Функция для чтения изображения + извлечения из него необходимых 
дескрипторов и гистрограммы.
-----------------------------------------------------------------------
Принимает: directory - имя папки, в которой находится изображение.
image - имя файла, содержащего необходимое изображение.
-----------------------------------------------------------------------
Возвращает: des - дескрипторы изображения.
hist - гистограмма изображения
-----------------------------------------------------------------------
'''


def LoadSamples(directory, image):
    img = cv2.imread(directory + image)
    des, hist = descriptors_and_hist(img)
    return des, hist


# обучающая выборка
# массив, содержащий кортежи дескрипторов и гистограмм всех изображений выборки
totalDescriptors = []
# массив, в который заносится имя класса каждого подаваемого на вход изображения
names = []
# словарь, содержащий пары "класс изображения: дескрипторы и гистограмма изображения"
samples = {}
directory = 'train/'
# общее число изображений в тренировочной выборке
totalSamplesNumber = 0
# рассматриваем все изображения внутри папки
for image in os.listdir(directory):
    # загружаем изображение и получаем его дескрипторы и гистограмму
    des_image = LoadSamples(directory, image)
    # класс изображения - имя файла до символа 'решетка'
    name = os.path.basename(directory + image)
    name = name[:name.index('#')]
    # в случае, если класс изображения уже есть в общем словаре, добавляем к нему только что найденные дескрипторы
    # иначе - под ключом имени класса создаем массив, содержащий пока только новые дескрипторы.
    if samples.get(name, 0) != 0:
        samples[name].append(des_image)
    else:
        samples[name] = [des_image]
    # добавляем в массив наименований классов класс нового изображения,
    # а в массив всех дескрипторов - только что найденные дескрипторы и гистограмму
    addNames(names, name, des_image)
    addDescriptors(totalDescriptors, des_image)
    totalSamplesNumber += 1
#  создание всех необходимых классификаторов
#  kmeans — для кластеризации
#  TfidfTransformer, MultinomialBN, AdaBoostClassifier — для классификации
#  параметры MiniBatchKMeans():
#  n_clusters- Количество кластеров для формирования, а также количество центроидов для генерации
#  random_state - Определяет генерацию случайных чисел для инициализации центроида и случайного переназначения
#  verbose - Режим детализации
kmeans = MiniBatchKMeans(n_clusters=CLUSTERS_NUMBER, random_state=CLUSTER_SEED, verbose=True)
#  Мера tf-idf: вес некоторого дескриптора пропорционален количеству его употребления в изображении,
#  и обратно пропорционален частоте его употребления в других документах изображениях.
_tfidf_orb = TfidfTransformer()
_tfidf_hist = TfidfTransformer()
_tfidf_knn = TfidfTransformer()
#  alpha - Параметр аддитивного (Лапласа/Лидстоуна) сглаживания
#  n_estimators - Максимальное количество оценок, при котором бустинг прекращается.
#  В случае идеальной подгонки процедура обучения останавливается досрочно.
clf_orb = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
clf_hist = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
clf_knn = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
#  n_neighbors - Количество "соседей"
knn = KNeighborsClassifier(n_neighbors=350)
# обучение моделей
kmeans.fit(totalDescriptors)
knn.fit(totalDescriptors, names)
# преобразование дескрипторов в кластеры посредством функции предсказания
# (в дальнейшем, для удобства, будем продолжать называть их дескрипторы)
kmeans_clus = numpy.array(kmeans.predict(totalDescriptors))
knn_clus = knn.predict(totalDescriptors)
# создание заготовок трех разреженных матриц
# матрицы, содержащие информацию о повторении дескрипторов-ORB для каждой фотографии
counts_orb = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER))
counts_knn = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER))
# матрица, содержащая информацию о повторении цветов на изображениях
counts_hist = lil_matrix((totalSamplesNumber, 256))
# переменные, необходимые для перебора всех примеров и дескрипторов
currentDescr = 0
currentSample = 0
for i in samples.values():
    currentDescr, currentSample = calculteCounts(i,  counts_orb, counts_hist, counts_knn,
                                                 kmeans_clus, knn_clus, currentDescr, currentSample)
# приведение разреженных матриц к удобному для работы виду
counts_orb = csr_matrix(counts_orb)
counts_hist = csr_matrix(counts_hist)
counts_knn = csr_matrix(counts_knn)
# преобразование матриц для обучения классификатора
tfidf_orb = _tfidf_orb.fit_transform(counts_orb)
tfidf_hist = _tfidf_hist.fit_transform(counts_hist)
tfidf_knn = _tfidf_knn.fit_transform(counts_knn)
# создание массива классов, где имя класса повторяется столько раз, сколько класс встречается в обучающей выборке
classes = []
for key, value in samples.items():
    classes += [key] * len(value)
# обучение байесовского классификатора
clf_orb.fit(tfidf_orb, classes)
clf_hist.fit(tfidf_hist, classes)
clf_knn.fit(tfidf_knn, classes)

# тестовая выборка для предсказаний
# действия, аналогичные обучению модели
directory = 'test/'
samples_test = []
names = []
totalDescriptors = []
images_test = []
for image in os.listdir(directory):
    images_test.append(cv2.imread(directory + image))
    name = os.path.basename(directory + image)
    names.append(name[:name.index('#')])
    s = LoadSamples(directory, image)
    samples_test.append(s)
    addDescriptors(totalDescriptors, s)
kmeans_clus = kmeans.predict(totalDescriptors)
clusters_knn = knn.predict(totalDescriptors)
counts_orb = lil_matrix((len(samples_test), CLUSTERS_NUMBER))
counts_hist = lil_matrix((len(samples_test), 256))
counts_knn = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER))
currentDescr = 0
currentSample = 0
currentDescr, currentSample = calculteCounts(samples_test,  counts_orb, counts_hist, counts_knn,
                                             kmeans_clus, clusters_knn, currentDescr, currentSample)
counts_orb = csr_matrix(counts_orb)
counts_hist = csr_matrix(counts_hist)
counts_knn = csr_matrix(counts_knn)
tfidf_kmeans = _tfidf_orb.transform(counts_orb)
tfidf_hist = _tfidf_hist.transform(counts_hist)
tfidf_knn = _tfidf_knn.fit_transform(counts_knn)
# определение весов каждого классификатора и предсказание
weights_kmeans = clf_orb.predict_log_proba(tfidf_kmeans)
weights_hist = clf_hist.predict_log_proba(tfidf_hist)
weights_knn = clf_knn.predict_log_proba(tfidf_knn)
# массив, содержащий предсказание модели, относительно поданных на вход изображений
predictions = []
# массив названий классов для предсказания
predict_classes = [i for i in samples.keys()]
# рассматриваем предсказания каждого классификатора по каждому изображению
for i in range(len(weights_kmeans)):
    # ищем индексы наибольших элементов в последовательностях предсказанных весов классификаторов
    index_i, max_value_i = max(enumerate(weights_kmeans[i]), key=lambda i_v: i_v[1])
    index_j, max_value_j = max(enumerate(weights_hist[i]), key=lambda i_v: i_v[1])
    index_k, max_value_k = max(enumerate(weights_knn[i]), key=lambda i_v: i_v[1])
    # составляем список индексов
    indx = [index_i, index_j, index_k]
    # считаем преобладающий номер позиции наибольшего из весов у каждого классификатора
    count_i = indx.count(index_i)
    count_j = indx.count(index_j)
    count_k = indx.count(index_k)
    counts = [count_i, count_j, count_k]
    # рассматриваем, какой из индексов встречается чаще всего среди предсказаний трех классификаторов,
    # класс под таким индексом в словаре и является предсказанием ансамбля
    # условные операторы расположены по приоритетности классификаторов
    if max(counts) == count_j:
        predictions.append(predict_classes[index_j])
    elif max(counts) == count_i:
        predictions.append(predict_classes[index_i])
    else:
        predictions.append(predict_classes[index_k])

if input("\n\nВывести каждое изображение с предсказанным моделью классом (y/n): ") == 'y':
    # вывод изображений тестовой выборки под заголовком с названием предсказанного класса
    for i in range(len(names)):
        plt.imshow(images_test[i]), plt.title(predictions[i]), plt.axis('off')
        plt.show()
print(f"\nИтоговый процент правильно предсказанных классов на {len(predictions)} изображений: "
      f"{100 * len([i for i in range(len(predictions)) if predictions[i] == names[i]]) / len(predictions)}%")
