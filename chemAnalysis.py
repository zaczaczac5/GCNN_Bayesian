count_TP = 484
print("True Positive:",count_TP)
count_FP = 779
print("False Positive:",count_FP)
count_FN = 472
print("False Negative:",count_FN)
count_TN = 2217
print("True Negative:",count_TN)
Accuracy = (count_TP + count_TN) / (count_TP + count_FP + count_FN + count_TN)
print("Accuracy:",Accuracy)
import math

MCC = (count_TP * count_TN - count_FP * count_FN) / math.sqrt(abs((count_TN
                                                                    + count_FP)
                                                                 * (count_TN + count_FN)
                                                                 * (count_TP + count_FP)
                                                              * (count_TP + count_FN)))

print("MCC:",MCC)
Specificity = count_TN / (count_TN + count_FP)
print("Specificity:",Specificity)
Precision = count_TP / (count_TP + count_FP)
print("Precision:",Precision)
# sensitivity
Recall = count_TP / (count_TP + count_FN)
print("Recall:",Recall)
# F1
Fmeasure = 2*(Precision * Recall) / (Recall + Recall)
print("Fmeasure:",Fmeasure)