import cv2
import sys
import numpy as np 
class WeedDetection:
    def __init__(self,img):
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.part_width = img.shape[1]//3
    
    def preprocess(self, img):
        '''
        Blur da imagem e converte em HSV
        '''
        kernel_size = 15
        img_blur = cv2.medianBlur(img, 15)
        img_hsv = cv2.cvtColor(img_blur,cv2.COLOR_BGR2HSV)
    
        return img_hsv

    def createMask(self, img_hsv):
        '''
        Cria mascara verde
        '''
        sensitivity = 20
        lower_bound = np.array([50 - sensitivity, 100, 60])
        upper_bound = np.array([50 + sensitivity, 255, 255])
        msk = cv2.inRange(img_hsv, lower_bound, upper_bound)

        return msk

    def transform(self, msk):
        '''
        erosao e dilatacao
        '''
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

        res_msk = cv2.morphologyEx(msk, cv2.MORPH_OPEN, kernel)
        res_msk = cv2.morphologyEx(res_msk, cv2.MORPH_CLOSE, kernel)

        return res_msk

    def calcPercentage(self, msk):
        '''
        porcentagem em branco
        '''
        height, width = msk.shape[:2]
        num_pixels = height * width
        count_white = cv2.countNonZero(msk)
        percent_white = (count_white/num_pixels) * 100
        percent_white = round(percent_white,2)

        return percent_white

    def weedPercentage(self, msk):
        '''
        Divide a mascara em 3 partes e calcula a porcentagem de verde
        '''
        left_part = msk[:,:self.part_width]
        mid_part = msk[:,self.part_width:2*self.part_width]
        right_part = msk[:,2*self.part_width:]
        left_percent = self.calcPercentage(left_part)
        mid_percent = self.calcPercentage(mid_part)
        right_percent = self.calcPercentage(right_part)

        return [left_percent, mid_percent, right_percent]

    def markPercentage(self, img, percentage):
        '''
        marca as porcentagens na imagem 
        '''
        part_width = self.width//3

        font = cv2.FONT_HERSHEY_SIMPLEX

        for i in range(3):
            cv2.putText(img, str(percentage[i]) + "%", (int(part_width*(i + 0.34)), self.height//2), font, 1, (0,0,255), 3, cv2.LINE_AA)

        return img

def main():

    cli_args = sys.argv[1:]

    if len(cli_args) != 1:
        print("python segmentation.py image_path")
        sys.exit(1)

    img_path = cli_args[0]
    img = cv2.imread(img_path)
    print(img.shape)
    img_resize = cv2.resize(img, (800,500))

    wd = WeedDetection(img)

    img_hsv = wd.preprocess(img)

    msk1 = wd.createMask(img_hsv)

    blurred = cv2.medianBlur(img, 21)
    
    edges = cv2.Canny(blurred, 50, 150)
    
    msk = wd.transform(msk1)

    percentage = wd.weedPercentage(msk)
    res = wd.markPercentage(img_resize, percentage)
    res_msk = cv2.bitwise_and(img,img,mask = msk)

    cv2.imshow('Res',res)
    cv2.imshow('Mask', res_msk)
    cv2.imshow('Frame',img)

    cv2.imshow('hsv',img_hsv)
    cv2.imshow('msk1',msk1)
    cv2.imshow('msk2',msk)
    cv2.imshow('blurred',blurred)

    cv2.imshow('blurred',blurred)
    cv2.imshow('edges',edges)

    
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
