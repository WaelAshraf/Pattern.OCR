import cv2
import numpy as np
# from commonfunctions import *
from scipy.stats import iqr
from scipy import stats
# from collections import namedtupl
from dataclasses import dataclass

def FindBaselineIndex(line): #Alg. 4
    HP = []
    PV = []
    BaseLineIndex = 0
    thresh,thresh_img = cv2.threshold(line,127,255,cv2.THRESH_BINARY_INV)
    thresh_img = np.asarray(thresh_img)
    thresh_img = line

    HP = np.sum(thresh_img, axis = 1)
    PV_Indices = (HP > np.roll(HP,1)) & (HP > np.roll(HP,-1))
    for i in range(len(PV_Indices)):
        if PV_Indices[i] == True:
            PV.append(HP[i])
    #print(PV)
    MAX = max(PV)
    for i in range(len(HP)):
        if HP[i] == MAX:
            BaseLineIndex = i
    # print(BaseLineIndex)
    # cv2.line(thresh_img, (0, BaseLineIndex), (thresh_img.shape[1], BaseLineIndex), (255,255,255), 1)
    # cv2.imshow('binary',thresh_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return BaseLineIndex
    
    
def FindingMaxTrans(Line, BaseLineIndex): #Alg. 5
    MaxTrans = 0
    MaxTransIndex = BaseLineIndex
    i=BaseLineIndex
    while i > 0:
        CurrTrans = 0
        Flag = 0
        j=0
        while j < Line.shape[1]:
            if Line[i, j] == 1 and Flag == 0:
                CurrTrans += 1
                Flag = 1
            if Line[i, j] != 1 and Flag == 1:
                Flag = 0
            j += 1

        if CurrTrans >= MaxTrans:
            MaxTrans = CurrTrans
            MaxTransIndex = i
        i -= 1
    
    # cv2.line(Line, (0, MaxTransIndex), (Line.shape[1], MaxTransIndex), (50,100,150), 1)
    # cv2.imshow('binary',Line)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return MaxTransIndex

def getVerticalProjectionProfile(image):
    vertical_projection = np.sum(image, axis = 0) 
    return vertical_projection 


@dataclass
class SeparationRegions:
    StartIndex: int=0
    EndIndex: int=0
    CutIndex: int=0

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]

#def CutPointIdentification(Line,Word,MTI): #Alg. 6
def CutPointIdentification(Word,MTI): #Alg. 6 ACCORDING TO THE PSEUDO CODE
    Flag=0
    #LineImage=cv2.imread(Line)
    VP=getVerticalProjectionProfile(Word)
    #MFV = stats.mode(VP)
    VPList = VP.tolist() #to be able to get the MFV
    Beginindex=0#ka2eni bashel el goz2 el black eli 3la el edges fl sora 3ashan ageb mode value mazbota
    EndIndex=len(VPList)
    for i in VPList:
        if i ==0:
            Beginindex+=1
        else:
            break
    for j in range(-1,-30,-1):
        if VPList[j]==0:
            EndIndex-=1
        else:
            break


    i=1
    VPListNew = VPList[Beginindex:EndIndex]
    MFV = max(set(VPListNew), key = VPListNew.count) 
    OutputSeparationRegions= []
    SRAppendFlag=False #initialize but do not append
    while i < Word.shape[1] :
        if SRAppendFlag == False:
            SR = SeparationRegions()
            SRAppendFlag = True
        if Word[MTI,i] == 1 and Word[MTI,i+1] == 0 and Flag == 0 : #CALCULATE END INDEX
            SR.EndIndex = i
            Flag = 1
        if i == (Word.shape[1]-1):
            break
        if Word[MTI,i] == 0 and Word[MTI,i+1] == 1 and Flag == 1 : #CALCULATE START AND CUT INDEX
            SR.StartIndex = i+1
            MidIndex = ( SR.EndIndex + SR.StartIndex )/2
            MidIndex = int(MidIndex)
            IndexesEqualZero = np.where(VP == 0)
            IndexesEqualZero = np.asarray(IndexesEqualZero)
            IndexesEqualZero = IndexesEqualZero.tolist()
            IndexesEqualZero = IndexesEqualZero[0]
            IndexesEqualZero = np.array(IndexesEqualZero)
            #print(IndexesEqualZero)
            IndexesCorrect= IndexesEqualZero [ (IndexesEqualZero < SR.StartIndex) & (IndexesEqualZero > SR.EndIndex)] #condition shall be reversed like this
            #IndexesCorrect = IndexesEqualZero[ mask ]
            #print(IndexesEqualZero [ (IndexesEqualZero < SR.StartIndex) & (IndexesEqualZero > SR.EndIndex)])

            IndexesLessThanMFVAndEnd = np.where( (VP <= MFV) )
            IndexesLessThanMFVAndEnd = np.asarray(IndexesLessThanMFVAndEnd)
            IndexesLessThanMFVAndEnd=IndexesLessThanMFVAndEnd.tolist()
            IndexesLessThanMFVAndEnd = IndexesLessThanMFVAndEnd[0]
            IndexesLessThanMFVAndEnd = np.array(IndexesLessThanMFVAndEnd)
            IndexesLessThanMFVAndEnd = IndexesLessThanMFVAndEnd [ (IndexesLessThanMFVAndEnd > SR.EndIndex) & (IndexesLessThanMFVAndEnd < MidIndex)  ]
            #IndexesLessThanMFVAndEnd.append(2)

            IndexesLessThanMFVAndStartAndEnd = np.where( (VP <= MFV) )
            IndexesLessThanMFVAndStartAndEnd = np.asarray(IndexesLessThanMFVAndStartAndEnd)
            IndexesLessThanMFVAndStartAndEnd=IndexesLessThanMFVAndStartAndEnd.tolist()
            IndexesLessThanMFVAndStartAndEnd = IndexesLessThanMFVAndStartAndEnd[0]
            IndexesLessThanMFVAndStartAndEnd = np.array(IndexesLessThanMFVAndStartAndEnd)
            IndexesLessThanMFVAndStartAndEnd = IndexesLessThanMFVAndStartAndEnd [ (IndexesLessThanMFVAndStartAndEnd > SR.EndIndex) & (IndexesLessThanMFVAndStartAndEnd < SR.StartIndex) ]
            

            if len(IndexesCorrect) != 0: #neither connected nor overlapped characters
                SR.CutIndex = find_nearest(IndexesCorrect , MidIndex)

            elif VP[MidIndex] == MFV: #connected characters
                SR.CutIndex = MidIndex #line 19 on Alg.
            
            elif len(IndexesLessThanMFVAndEnd) != 0: 
                SR.CutIndex = find_nearest(IndexesLessThanMFVAndEnd , MidIndex)
            
            elif len(IndexesLessThanMFVAndStartAndEnd) != 0: #line 23
                SR.CutIndex = find_nearest(IndexesLessThanMFVAndStartAndEnd[1:] , MidIndex)
            else:
                SR.CutIndex = MidIndex
            
            if SRAppendFlag == True:
                OutputSeparationRegions.append(SR)
                SRAppendFlag = False
            Flag = 0
        i+=1
    return OutputSeparationRegions


im = cv2.imread('images/30.png', cv2.IMREAD_GRAYSCALE)
ret, thresh = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY_INV)
#cv2.imshow('str',thresh/255)   
BaselineIndex = FindBaselineIndex(thresh)
print(BaselineIndex)
# for i in range(25):
#     for j in range(673):
#         print(thresh[i,j])
MaxTransitionIndex = FindingMaxTrans(thresh/255, BaselineIndex)
print("max")
print(MaxTransitionIndex)

SeparationRegions = CutPointIdentification(thresh/255, MaxTransitionIndex)
print("Seeing Cut Point Identification")
for SR in SeparationRegions:
    cv2.line(thresh, (BaselineIndex, SR.StartIndex), (BaselineIndex, SR.StartIndex+1), (0, 20, 200), 10)
    print(SR.StartIndex)
    print(SR.EndIndex)
    print(SR.CutIndex)
    print("*********")
cv2.imshow('Window', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()