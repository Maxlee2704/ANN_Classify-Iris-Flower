# Khai báo thư viện
import cv2
import numpy as np
from PIL import Image,ImageEnhance

# Các hàm tiền xử lý ảnh:
# Scale ảnh theo một tỉ lệ nào đó

def cscale(img,rate):
    """
        Được sử dụng đê scale ảnh
        Input: array
        Output: array
    """
    w = int(img.shape[0]* rate)
    h = int(img.shape[1] * rate)
    img_scale = cv2.resize(img,(256,256),interpolation=cv2.INTER_LINEAR)
    return img_scale


def cmode(img,mode):
    """
        Được sử dụng để chuyển mode ảnh
        Input: array
        Output: array
    """
    if mode=='HSV':
        img_mode = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    if mode=='Gray':
        img_mode = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    else: img_mode = img
    return img_mode

def cbrightness(img,factor):
    """
        Được sử dụng để thay đổi độ sáng
        Input: image object
        Output: image object
    """
    enhancer = ImageEnhance.Brightness(img)
    img_bright = enhancer.enhance(factor)
    return img_bright

def ccontrast(img,factor):
    """
        Được sử dụng để thay đổi độ tương phản
        Input: image object
        Output: image object
    """
    enhancer = ImageEnhance.Contrast(img)
    img_contrast = enhancer.enhance(factor)
    return img_contrast

def csaturation(img,factor):
    """
        Được sử dụng để thay đổi độ bảo hòa
        Input: image object
        Output: image object
    """
    enhancer = ImageEnhance.Color(img)
    img_saturation = enhancer.enhance(factor)
    return img_saturation





def detect(img_orginal, img):
    # Cân chỉnh mức xám của anh
    alpha = 255/np.max(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            img[i,j]= img[i,j]* alpha

    # Sử dụng Sobel để phát hiện biên
    grad_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=15,  borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=15,  borderType=cv2.BORDER_DEFAULT)
    # Làm nổi ảnh
    img = img + grad_x + grad_y


    # Lọc nhiễu ảnh
    img = cv2.bilateralFilter(img,7,15,11)
    cv2.imshow('Filter Img',img)
    cv2.waitKey()


    # Phân ngưỡng thích nghi để giữ lại phần quan trọng
    img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,5,-1)
    # Loại bỏ những vùng ảnh có kích thước nhỏ (Nhiễu sau khi phân ngưỡng)
    # Sau đó, dilate để trả lại kích thước cũ
    img = cv2.erode(img,(3,3),iterations=2)
    img = cv2.dilate(img,(5,5),iterations=4)
    cv2.imshow('Thresold',img)
    cv2.waitKey()

    # Tìm đường biên (Contour) của ảnh
    cont,_= cv2.findContours(img, mode=cv2.RETR_EXTERNAL, method= cv2.CHAIN_APPROX_NONE)
    cont = sorted(cont, key=cv2.contourArea, reverse=True)[:5]
    error_num = 0
    result = []
    # Bounding box
    for c in cont:
        if cv2.contourArea(c)>20:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img_orginal, (x, y), (x + w, y + h), (0, 0, 255), 1)
            error_num +=1
            result.append([x,y,w,h])

    print('Detect '+str(error_num)+' error : ',result)

    cv2.imshow('Processed Img',img_orginal)
    cv2.waitKey()

if __name__== "__main__":
    path = input("Link image: ")
    img_object = Image.open(path)

    print('Mode: ',img_object.mode)

    bright = input("Bright: ")
    img_object = cbrightness(img_object, float(bright))

    contrast = input("Contrast: ")
    img_object = ccontrast(img_object, float(contrast))


    saturation = input("Saturation: ")
    img_object = csaturation(img_object, float(saturation))

    img_orginal = np.array(img_object)
    rate = input("Rate: ")
    img_orginal= cscale(img_orginal,rate)
    img = img_orginal

    #img = cv2.cvtColor(img_orginal,cv2.COLOR_BGR2GRAY)

    detect(img_orginal, img)
