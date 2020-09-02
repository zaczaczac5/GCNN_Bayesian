#this file calculates the stats for each chemical class
Exp=[0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1 ]
Pred=[ 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0  ]
True_positive=0
False_postive=0
True_negative=0
False_negative=0
for i in range(len(Exp)):
        if(Exp[i]==Pred[i] and Exp[i]==1):
            True_positive+=1
        if (Exp[i] != Pred[i] and Exp[i] == 0):
            False_postive+= 1
        if (Exp[i] == Pred[i] and Exp[i] == 0):
            True_negative+= 1
        if (Exp[i] != Pred[i] and Exp[i] == 1):
            False_negative+= 1
count_TP = True_positive
print(count_TP)
count_FP = False_postive
print(count_FP)
count_FN = False_negative
print(count_FN)
count_TN = True_negative
print(count_TN)
Accuracy = (count_TP + count_TN) / (count_TP + count_FP + count_FN + count_TN)
print(Accuracy)
import math

MCC = (count_TP * count_TN - count_FP * count_FN) / math.sqrt(abs((count_TN
                                                                    + count_FP)
                                                                 * (count_TN + count_FN)
                                                                 * (count_TP + count_FP)
                                                              * (count_TP + count_FN)))

print(MCC)
Specificity = count_TN / (count_TN + count_FP)
print(Specificity)
Precision = count_TP / (count_TP + count_FP)
print(Precision)
# sensitivity
Recall = count_TP / (count_TP + count_FN)
print(Recall)
# F1
Fmeasure = 2*(Precision * Recall) / (Recall + Recall)
print(Fmeasure)

#Alcohols
#Benzidines
#ethers


