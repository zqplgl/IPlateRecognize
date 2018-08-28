#coding=utf-8
import _VehiclePlate as pr
import cv2
import os

class IPlateRecognize:
    def __init__(self,modelDir,gpu_id=0):
        self.__detector = pr.PlateDetector(modelDir,gpu_id)
        self.__gpu_id = gpu_id

    def detect(self,im):
        confidence_threshold = 0.7
        result =[]
        plateinfos = self.__detector.detect(im,im.shape[1],im.shape[0],confidence_threshold)
        for plateinfo in plateinfos:
            temp = {} 
            temp["license"] = plateinfo.license
            temp['color'] = plateinfo.color
            temp['zone'] = (plateinfo.zone.x,plateinfo.zone.y,plateinfo.zone.x+plateinfo.zone.w,plateinfo.zone.y+plateinfo.zone.h)
            temp['score'] = plateinfo.score

            result.append(temp)

        sorted(result,key=lambda obj:obj['score'])

        if len(result):
            return result[0]
        else:
            return None


def run():
    modelDir = r"/home/zqp/install_lib/models"
    plateDetector = IPlateRecognize(modelDir,0)
    picDir = "/home/zqp/testimage/image/"

    for picName in os.listdir(picDir):
        print picName
        
        im = cv2.imread(picDir+picName)
        result = plateDetector.detect(im)

        cv2.imshow("im",im)
        if result:
            print result["license"],"\t",result["color"],"\t",result["zone"],"\t",result["score"]

        if cv2.waitKey(0)==27:
            break


if __name__=="__main__":
    run()

