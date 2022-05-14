import os
import numpy
import cv2
from sklearn.cluster import MiniBatchKMeans
from scipy.sparse import lil_matrix, csr_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from PIL import Image

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
orient = 9
cells_Per_Block = 2
pixels_Per_Cell = 16

# находим дескрипторы
def descriptors(img):
    #  меняем размер файла, тк на файлах с большим разрешением дескрипторы считаются дольше
    if img.shape[1] > 1000:
        cf = 1000.0 / img.shape[1]
        newSize = (int(cf * img.shape[0]), int(cf * img.shape[1]), img.shape[2])
        img.resize(newSize, refcheck=False)
    # создание orb объекта
    orb = cv2.ORB_create()
    # нахождение ключевых точек и дескрипторов
    kp, des = orb.detectAndCompute(img, None)
    hist = getColorHist(img)
    # код для проверки нахождения контрольных точек алгоритмом:
    # img2 = cv2.drawKeypoints(img, kp, None, color=(0, 255, 0), flags=0)
    # plt.imshow(img2, cmap='gray'), plt.title('ORB'), plt.axis('off')
    # plt.show()
    return des, hist

# считываем цветовую гистограмму для второго классификатора
def getColorHist(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    dist = cv2.calcHist([hsv], [0], None, [256], [0, 256])
    return dist


def addDescriptors(totalDescriptors, sample):
    for descriptor in sample[0]:
        totalDescriptors.append(descriptor)


def addNames(names, name, sample):
    for _ in sample[0]:
        names.append(name)


def calculteCounts(samples, counts, counts1, counts2, clusters, clusters1, currentDescr, currentSample):
    # проходимся по всем наборам для каждой пикчи: дескрипторы + цв. гист.
    for s in samples:
        # текущее количество повторений каждого дескриптора
        currentCounts = {}
        curva = {}
        # смотрим дескрипторы для данной картинки
        for _ in s[0]:
            # в словарик под ключом, обозначающим кластер определенного дескриптора
            # прибавляем единичку, если такой дескриптор уже встречался.
            # иначе просто добавляем в словарь информацию о наличии такого дескриптора
            # грубо говоря, считаем количество кластеров каждого дескриптора
            currentCounts[clusters[currentDescr]] = currentCounts.get(clusters[currentDescr], 0) + 1
            curva[clusters1[currentDescr]] = curva.get(clusters1[currentDescr], 0) + 1
            # увеличиваем количество дескрипторов, чтобы перейти к следующему дескриптору в массиве кластеров
            currentDescr += 1
        # рассматриваем пары вида "дескриптор: кол-во его повторений"
        for clu, cnt in currentCounts.items():
            # обновляем информацию внутри матриц о встречаемости каждого кластера
            counts[currentSample, clu] = cnt
        for i in range(len(curva)):
            counts2[currentSample, i] = curva[list(curva.keys())[i]]
        for i, histCnt in enumerate(s[1]):
            # обновляется информация в каунтс1, добавляя туда только первый
            # (т.е. там, где нет нулей, вроде) элемент гист.
            counts1[currentSample, i] = histCnt[0]
            if i == 255:
                break
        # количество пройденных примеров + 1
        currentSample += 1
    # возвращаем количество встреченных дескрипторов и количество пройденных примеров
    return currentDescr, currentSample


def LoadSamples(directory, image):
    img = cv2.imread(directory + image)
    des, hist = descriptors(img)
    return des, hist

# загрузка обучающей выборки и нахождение дескрипторов
# положительные и отрицательные примеры

totalDescriptors = []
names = []
samples = {}
directory = 'images/'
for image in os.listdir(directory):
    s = LoadSamples(directory, image)
    name = os.path.basename(directory + image)
    name = name[:name.index('#')]
    # тк у нас все еще попадаются ломаные кортинки, создаем такой условный оператор
    if s != None:
        if samples.get(name, 0) != 0:
            samples[name].append(s)
        else:
            samples[name] = [s]
        addNames(names, name, s)
        addDescriptors(totalDescriptors, s)
    else:
        print("ERROR ON", image)
#  создание всех необходимых классификаторов
#  kmeans — для кластеризации
#  TfidfTransformer, MultinomialBN, AdaBoostClassifier — для классификации
kmeans = MiniBatchKMeans(n_clusters=CLUSTERS_NUMBER, random_state=CLUSTER_SEED, verbose=True)
_tfidf = TfidfTransformer()
_tfidf1 = TfidfTransformer()
_tfidf2 = TfidfTransformer()
clf = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
clf1 = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
clf2 = AdaBoostClassifier(MultinomialNB(alpha=BAYES_ALPHA), n_estimators=ADA_BOOST_ESTIMATORS)
model = KNeighborsClassifier(n_neighbors=100)
#  обучение модели
kmeans.fit(totalDescriptors)
model.fit(totalDescriptors, names)
# список кластеров с соответствием дескрипторам
clusters = kmeans.predict(totalDescriptors)
clusters = numpy.array(clusters)
totalSamplesNumber = len(samples)
# создание двух матриц, подсчет частоты встречаемости
# сколько раз встретились дескрипторы из каждого кластера для каждой картинки
counts = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER))
# сколько раз встретился каждый цвет для каждой картинки
counts1 = lil_matrix((totalSamplesNumber, 256))
counts2 = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER))
currentDescr = 0
currentSample = 0
for i in samples.values():
    currentDescr, currentSample = calculteCounts(i,  counts, counts1, counts2, clusters, names, currentDescr, currentSample)
counts = csr_matrix(counts)
counts1 = csr_matrix(counts1)
counts2 = csr_matrix(counts2)
# преобразование матриц
tfidf = _tfidf.fit_transform(counts)
tfidf1 = _tfidf1.fit_transform(counts1)
tfidf3 = _tfidf2.fit_transform(counts2)
classes = []
for key, value in samples.items():
    classes += [key] * len(value)
# обучение байесовского классификатора
clf.fit(tfidf, classes)
clf1.fit(tfidf1, classes)
clf2.fit(tfidf3, classes)

# обработка изображения

# функция считывания изображения
def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# изначальное тестовое изображение, на котором будут рисоваться контуры
iMg = cv2.imread('photo_2022-05-11_23-50-24.jpg')
# то же самое вспомогательное изображение, которое будет подвергаться редактированию
image = cv2.imread('photo_2022-05-11_23-50-24.jpg')

# наложение фильтров на изображение
hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

low = numpy.array((59, 119, 17), numpy.uint8)
high = numpy.array((79, 255, 255), numpy.uint8)
curr_mask = cv2.inRange(hsv_img, low, high)
hsv_img[curr_mask > 0] = ([75, 255, 200])

hsv_img = 255 - hsv_img  # neg = (L-1) - img

RGB_again = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
gray = cv2.cvtColor(RGB_again, cv2.COLOR_RGB2GRAY)

gray = cv2.GaussianBlur(gray, (3, 3), 0)
gray = cv2.bilateralFilter(gray,9,75,75)

ret, threshold = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

# находим и рисуем координаты контуров на основе отредактированного изображения
contours, hierarchy = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (0, 0, 255), 3)

# словарь координат фигур
figures_coordinates = {}
# временные названия контуров
k = [str(j) for j in range(100)]
j = 0
# проходимся по координатам всех найденных контуров
for i in range(len(contours)):
    # вписываем координаты найденной фигуры в прямоугольник
    x, y, w, h = cv2.boundingRect(contours[i])
    # проверка на случай, если нашлись какие-то побочные, мелкие контуры
    if w > 20 and h > 30:
        # вырезаем фигуру по координатам
        img_crop = image[y - 15:y + h + 15, x - 15:x + w + 15]
        figures_name = k[j]
        j += 1
        # добавляем найденный контур к остальным
        figures_coordinates[figures_name] = (x - 15, y - 15, x + w + 15, y + h + 15)

# cмотрим найденные контуры
for key, value in figures_coordinates.items():
    # создаем побочное изображение, идентичное прежнему, оттуда вырезаем кусочек по нынешнему контуру
    im = Image.open('photo_2022-05-11_23-50-24.jpg')
    im_crop = im.crop((value[0], value[1], value[2], value[3]))
    # сохраняем найденный кусочек в папочку test под соответствующим ему номером
    # (напомню, что все ключи в словаре также являются цифрами)
    im_crop.save(f'test/{key}.jpg', quality=95)
    # im_crop.show()

# тестовая выборка для предсказаний
# действия, аналогичные обучению модели
# считываем ранее найденные куски изображений.
# предполагается, что именно них изображены нужные предметы
directory = 'test/'
samples1 = []
names = []
totalDescriptors = []
for image in os.listdir(directory):
    names.append(image[:image.index('.')])
    s = LoadSamples(directory, image)
    samples1.append(s)
    addDescriptors(totalDescriptors, s)
clusters = kmeans.predict(totalDescriptors)
clusters1 = model.predict(totalDescriptors)
counts = lil_matrix((len(samples1), CLUSTERS_NUMBER))
counts1 = lil_matrix((len(samples1), 256))
counts2 = lil_matrix((totalSamplesNumber, CLUSTERS_NUMBER))
currentDescr = 0
currentSample = 0
currentDescr, currentSample = calculteCounts(samples1,  counts, counts1, counts2, clusters, clusters1, currentDescr, currentSample)
counts = csr_matrix(counts)
counts1 = csr_matrix(counts1)
counts2 = csr_matrix(counts2)
tfidf = _tfidf.transform(counts)
tfidf1 = _tfidf1.transform(counts1)
tfidf2 = _tfidf2.fit_transform(counts2)
# определение весов каждого классификатора и предсказание
weights = clf.predict_log_proba(tfidf)
weights1 = clf1.predict_log_proba(tfidf1)
weights2 = clf2.predict_log_proba(tfidf2)
predictions = []
figures = [i for i in samples.keys()]
figures1 = [j for j in samples.keys()]
figures2 = [k for k in samples.keys()]
for i in range(len(weights)):
    f = 0
    index_i, max_value_i = max(enumerate(weights[i]), key=lambda i_v: i_v[1])
    index_j, max_value_j = max(enumerate(weights1[i]), key=lambda i_v: i_v[1])
    index_k, max_value_k = max(enumerate(weights2[i]), key=lambda i_v: i_v[1])
    indx = []
    indx.append(index_i)
    indx.append(index_j)
    indx.append(index_k)
    count = indx.count(index_i)
    count1 = indx.count(index_j)
    count2 = indx.count(index_k)
    counts = []
    counts.append(count)
    counts.append(count1)
    counts.append(count2)
    if max(counts) == count:
        predictions.append(figures[index_i])
    elif max(counts) == count1:
        predictions.append(figures1[index_j])
    else:
        predictions.append(figures2[index_k])

# переносим ранее найденные контуры в новый словарь, теперь уже ключами являются предсказанные моделью названия классов.
# тк предсказание происходит согласно тому порядку, который заявлен в папке, а все изображения у нас названы
# порядковыми номерами, то несложно догадаться, что, например, первому изображению соответствует первое предсказание
j = 0
res = {}
for key, values in figures_coordinates.items():
    res[predictions[j]] = values
    j += 1

# картинка, которая будет выводиться
img_ = cv2.imread('photo_2022-05-11_23-50-24.jpg')
for key, value in res.items():
    # наносим на изначальное, не обработанное изображение контур в виде прямоугольника
    rec = cv2.rectangle(img_, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
    # подписываем этот контур согласно ключу в словаре (предсказанному ему классу)
    cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

# выводим результат
viewImage(img_)
