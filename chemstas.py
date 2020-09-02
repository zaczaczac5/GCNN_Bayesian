structure = open("C:/Users/Zac Hung/Desktop/masterthesis/experiment/tobeworked", "r")
lines = [line.rstrip() for line in structure]

structure.close()
smiles = open("C:/Users/Zac Hung/Desktop/masterthesis/experiment/tobeworkedS", "r")
lines1 = [line.rstrip() for line in smiles]

smiles.close()


experimental_value = open("C:/Users/Zac Hung/Desktop/masterthesis/experiment/tobeworkedE", "r")
lines2 = [line.rstrip() for line in experimental_value]

experimental_value.close()

pred_value = open("C:/Users/Zac Hung/Desktop/masterthesis/experiment/tobeworkedP", "r")
lines3 = [line.rstrip() for line in pred_value]

pred_value.close()
stuff = open("C:/Users/Zac Hung/Desktop/masterthesis/experiment/structD.txt", 'w')
#count_substructure=0

count=0
count_total_tox=0
count_total_nonT=0
count_total_neg=0
count_total_falseneg=0
print(len(lines))
print(len(lines1))

for i in range(len(lines)):
    count_tox = 0
    count_nonT = 0
    count_neg = 0
    count_false_neg = 0
    pos = 0
    neg = 0
    neither = 0
    for j in range(len(lines1)):

        if (((lines1[j]).find(lines[i]))!= -1):

            #print(lines1[j])
            #print(lines[i])
            #count_substructure+=1
            if lines2[j]=='1' and lines3[i]=='1':
                count_tox+=1
                pos=1
            if lines2[j] == '0' and lines3[i] == '1':
                count_nonT +=1
                pos=1
            if lines2[j]=='0' and lines3[i]=='0':
                count_neg+=1
                neg = 1
            if lines2[j] == '1' and lines3[i] == '0':
                count_false_neg +=1
                neg = 1



    if(pos==1):
        tmp=(str(lines[i])+","+str(count_tox/(count_tox+count_nonT))+","+str(count_nonT/(count_tox+count_nonT))+","+"0.0"+","+"0.0"
                         +"\n")
        stuff.writelines(tmp)
        count_total_tox+=count_tox/(count_tox+count_nonT)
        count_total_nonT += count_nonT / (count_tox + count_nonT)
        pos = 0
    elif(neg == 1 ):
        tmp=(str(lines[i])+","+"0.0"+","+"0.0"+","+str(count_neg / (count_neg + count_false_neg)) + "," + str(
            count_false_neg / (count_neg + count_false_neg))+"\n")
        stuff.writelines(tmp)
        count_total_neg += count_neg / (count_neg + count_false_neg)
        count_total_falseneg += count_false_neg / (count_neg + count_false_neg)
        neg = 0
    else :
        stuff.writelines(str(lines[i])+",none,none,none,none"+"\n")

print("true positive",str((count_total_tox)/(count_total_tox+count_total_nonT)))
print("true negative",str((count_total_neg)/(count_total_neg+count_total_falseneg)))

    #stuff.writelines(str(count_tox)+"\t"+str(count_nonT)+"\n")