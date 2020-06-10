import flask
import werkzeug
import time
import cv2
import numpy as np
import base64

def mid_circle(hough,img):
    distance_x = 100
    distance_y = 100
    index = 0
    h = int(img.shape[0])
    w = int(img.shape[1])
    x_center = w/2
    y_center = h/2
    #print('center x,y = ',x_center,',',y_center)
    for i in range(10):
        if i < len(hough):
            r = int(hough[i][2])
            x = int(hough[i][0]) - r
            y = int(hough[i][1]) - r

            if y > 5 and x > 5 and y+r*2 < h and x+r*2 < w:
                x = x + r
                y = y + r
                #print(i,'. x,y=',x,',',y,'abs=',abs(x-x_center),',',abs(y-y_center))

                if abs(x-x_center) < distance_x and abs(y-y_center) < distance_y:
                    distance_x = abs(x-x_center)
                    distance_y = abs(x-x_center)
                    index = i            
    return index



app = flask.Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def handle_request():
    files_ids = list(flask.request.files)
    imagefile = flask.request.files[files_ids[0]]
    filename = werkzeug.utils.secure_filename(imagefile.filename)
    imagefile.save(filename)

    img = cv2.imread(filename)
    print('success get image!!!')
    
    #resize
    height = int(img.shape[0])
    width = int(img.shape[1])
    ratio = height/width
    width = 800
    height = int(800*ratio)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

    #Histogram Equalization
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    avg_color_per_row = np.average(img_gray, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    if avg_color < 85:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        #cv2.imshow('shv.jpg',img)
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
        #cv2.imshow('equ_shv.jpg',img)
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    #threshold
    ret,th = cv2.threshold(img,70,255,cv2.THRESH_BINARY)

    #Gray scale
    img_gray = cv2.cvtColor(th,cv2.COLOR_BGR2GRAY)

    #threshold
    ret,th = cv2.threshold(img_gray,70,255,cv2.THRESH_BINARY)

    #open
    kernel = np.ones((4,4),np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)

    #hough transform
    circles = cv2.HoughCircles(th,cv2.HOUGH_GRADIENT,1,50,param1=30,param2=15,minRadius=0,maxRadius=250)#60
    index = 0
    
    if circles is None:
        print('Not found the eye')
    else:
        circles = np.uint16(np.around(circles))
        index = mid_circle(circles[0],img)
        
        c = 0
        #for i in circles[0,:]:
            #cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            #cv2.putText(img,''+str(c), (i[0],i[1]), cv2.FONT_HERSHEY_COMPLEX, 1, 255)
            #c += 1
        #cv2.circle(img,(circles[0][index][0],circles[0][index][1]),circles[0][index][2],(0,0,255),2)

    #cv2.imshow('find_cornea.jpg',img)
    result1=''
    result2=''

    if circles is not None:
        x = int(circles[0][index][0])
        y = int(circles[0][index][1])
        r = int(circles[0][index][2])
        x = x-r
        y = y-r

        # crop image as a square
        img = img[y:y+r*2+2, x:x+r*2+2]
        result = img
        
        # create a mask
        mask = np.full((img.shape[0], img.shape[1]), 0, dtype=np.uint8) 
        # create circle mask, center, radius, fill color, size of the border
        cv2.circle(mask,(r,r), r, (255,255,255),-1)
        # get only the inside pixels
        fg = cv2.bitwise_or(img, img, mask=mask)

        mask = cv2.bitwise_not(mask)
        background = np.full(img.shape, 255, dtype=np.uint8)
        bk = cv2.bitwise_or(background, background, mask=mask)
        cornea = cv2.bitwise_or(fg, bk)

        #cv2.imshow('cornea.jpg',cornea)

        #find pterygium
        cornea_gray = cv2.cvtColor(cornea,cv2.COLOR_BGR2GRAY)
        cornea_gray = cv2.cvtColor(cornea_gray,cv2.COLOR_GRAY2BGR)
        #cv2.imshow('cornea_gray.jpg',cornea_gray)
        avg_color_per_row = np.average(cornea_gray, axis=0)
        avg_color = np.average(avg_color_per_row, axis=0)
        intensity = avg_color[0]*0.60
        #print (intensity)
        
        imgthreshold = cv2.inRange(cornea_gray, (intensity,intensity,intensity), (185,185,185))#185
        imgthreshold = cv2.bitwise_not(imgthreshold)
        #cv2.imshow('color_threshold.jpg',imgthreshold)

        #Closing
        kernel = np.ones((8,8),np.uint8)
        pterygium = cv2.morphologyEx(imgthreshold, cv2.MORPH_CLOSE, kernel)
        
        #cv2.imshow('close.jpg',pterygium)


        #Fill hole
        pass1 = np.full(pterygium.shape, 255, np.uint8)
        im_inv = cv2.bitwise_not(pterygium)
        mask1 = cv2.copyMakeBorder(im_inv, 1, 1, 1, 1, cv2.BORDER_CONSTANT, 0)
        _, pass1, _, _ = cv2.floodFill(pass1, mask1, (0,0), 0, 0, 0, 4)
        pterygium = cv2.bitwise_not(pass1)
        #cv2.imshow('fill_pterygium.jpg',pterygium)

        #Analyze
        area_cornea = (22/7)*(r*r)
        width_cornea = r*2
        
        contours, _ = cv2.findContours(pterygium,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        count = 0
        have_pterygium = 0
        area_pterygium = 0
        width_pterygium = 0
        height = int(cornea.shape[0])
        width = int(cornea.shape[1])
        for c in contours:
            area = cv2.contourArea(c)
            x,y,w,h = cv2.boundingRect(c)
            height_ratio = (y+(h/2)) / height *100
            width_ratio_left = x / width *100
            width_ratio_right = (x+w) / width *100
            check_position = 0
            if width_ratio_left < 2 or width_ratio_right > 98:
                check_position = 1
            if count >= 1 and area > 350 and height_ratio > 30 and height_ratio < 70 and check_position:
                #print('left=',width_ratio_left,'  right=',width_ratio_right)
                area_pterygium = area_pterygium + area
                width_pterygium = width_pterygium + w
                result = cv2.drawContours(result, contours, count, (230,200,70), 2)
                have_pterygium = 1
            count = count+1
        if have_pterygium == 0:
            print('Not found the pterygium')
        else:
            result1 = (area_pterygium / area_cornea)*100
            result2 = (width_pterygium / width_cornea)*100
            print('Ratio of pterygium area with cornea area: ',round(result1, 2),'%')
            print('Ratio of pterygium width with cornea width: ',round(result2, 2),'%')
        #result = np.hstack((cornea,result))
        #cv2.imshow('draw_pterygium.jpg',result)
        #cv2.imwrite('result_'+img_name,result)

    #resize before return
    height = int(result.shape[0])
    width = int(result.shape[1])
    ratio = height/width
    width = 100
    height = int(100*ratio)
    dim = (width, height)
    new_img = cv2.resize(result, dim, interpolation = cv2.INTER_AREA)

    
    #for return
    retval, buffer = cv2.imencode('.jpg', new_img)
    img_text = base64.b64encode(buffer)
    text = img_text.decode("utf-8")

    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    if result1 != '' and result2 != '':
        result1 = round(result1, 2)
        result2 = round(result2, 2)
    return str(result1)+'@'+str(result2)+'@'+text

app.run(host="0.0.0.0", port=5000, debug=True)


