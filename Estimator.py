import cv2 
import numpy as np
import math
from keras.models import load_model

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    CImg = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    _, binary = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    largest_contour = max(contours, key=cv2.contourArea)
    LArea=0
    for j in range(0,len(contours)):
        area = cv2.contourArea(contours[j])
        if(area>LArea):
            LArea = area
            LIndex = j
            BRect = cv2.boundingRect(contours[j])
    cv2.drawContours( CImg, contours,LIndex, ( 0, 255, 0 ), 5 )
    area = cv2.contourArea(largest_contour)
    
    return area,image,blurred,binary,CImg

def estimate_weight(area, Label):
    match Label:
        case "Organic":
            density = 0.05
        case "Metal":
            density = 0.007784
        case "Plastic":
            density = 0.0009
    real_area=area*(3.5**2)/100#zooming factor
    mass = real_area * density # write it
    estimated_weight = mass*9.8
    
    return estimated_weight,density,mass

def Sabry(image_path):
    Label = WasteClassifier(image_path)
    area,image,blurred,binary,CImg = preprocess_image(image_path)
    estimated_weight,density,mass = estimate_weight(area, Label)
    
    return estimated_weight, image, blurred, binary, CImg, Label,area,density,mass

















def AreaEstimator(i):
    img = cv2.imread(i)
    CImg = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)

    contours,ret = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #cv2.drawContours(CImg, contours, -1, (0,255,0), 5)
    #cv2.drawContours(thresh, contours, -1, (0,255,0), 5)
    #cv2.drawContours(gray, contours, -1, (0,255,0), 5)
    
    c = max(contours, key = cv2.contourArea)
    x,y,w,h = cv2.boundingRect(c)
    LArea=0
    for j in range(0,len(contours)):
        area = cv2.contourArea(contours[j])
        if(area>LArea):
            LArea = area
            LIndex = j
            BRect = cv2.boundingRect(contours[j])
   
    x,y,w,h = BRect
    print(LArea,w,h)
    cv2.drawContours( CImg, contours,LIndex, ( 0, 255, 0 ), 5 )
    cv2.imshow("s",CImg)
    depth = math.sqrt((w**2)+(h**2))
    return (LArea,img, gray, thresh, CImg,depth)
def WasteClassifier(i):
    img = cv2.imread(i)
    img = cv2.resize(img,(224,224))
    img = np.array(img)
    img = img / 255.0 # normalize the image
    img = img.reshape(1, 224, 224, 3) # reshape for prediction
    model = load_model("Waste35.h5")
    preds = model.predict(img)
    print(type(preds))
    print(preds)
    preds = preds.tolist()[0]
                    
    pred = preds.index(max(preds))
    print(pred)
    if pred == 0:
        label = 'Organic'
    elif pred == 1:
        label = 'Metal'
    else:
        label = 'Plastic'
    return label
def Pixel2M(area,depth):
    #38924.5 = 58.5cm
    # area * factor * depth / 100 *1000

    volume = area * depth * 0.0015029094 / 100 
    return volume
def AndalibEstimator(i):
    Label = WasteClassifier(i)
    Area, im, gr, thr, CI, depth = AreaEstimator(i)
    volume = Pixel2M(Area,depth)
    weight = 0
    density = 0
    match Label:
        case "Organic":
            density = 0.05
            mass = density * volume
            weight = mass * 9.8
        case "Metal":
            density = 0.007784
            mass =  density * volume
            weight = mass* 9.8
        case "Plastic":
            density = 0.0009
            mass = density * volume
            weight = mass * 9.8
        case _:
            print("Some kind of wierd behaviour happened")
    return weight, im, gr, thr, CI, Label,Area,depth,volume,density,mass



    