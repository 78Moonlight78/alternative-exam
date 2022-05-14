import cv2
import numpy
from PIL import Image

# функция открытия изображения
def viewImage(image):
    cv2.namedWindow('Display', cv2.WINDOW_NORMAL)
    cv2.imshow('Display', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# изначальное тестовое изображение, на котором будут рисоваться контуры
iMg = cv2.imread('your_photo.jpg')
# то же самое вспомогательное изображение, которое будет подвергаться редактированию
image = cv2.imread('your_photo.jpg')

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
    # наносим на изначальное, не обработанное изображение контур в виде прямоугольника
    rec = cv2.rectangle(iMg, (value[0], value[1]), (value[2], value[3]), (255, 255, 0), 2)
    # подписываем этот контур согласно ключу в словаре (предсказанному ему классу)
    cv2.putText(rec, key, (value[0], value[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
    # создаем побочное изображение, идентичное прежнему, оттуда вырезаем кусочек по нынешнему контуру
    im = Image.open('your_photo.jpg')
    im_crop = im.crop((value[0], value[1], value[2], value[3]))
    # сохраняем найденный кусочек в папочку test под соответствующим ему номером
    # (напомню, что все ключи в словаре также являются цифрами)
    im_crop.save(f'test/{key}.jpg', quality=95)
    # при необходимости, можно вывести вырезанный кусок
    # im_crop.show()

viewImage(iMg)