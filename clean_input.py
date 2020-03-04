filename="C:/Users/Zac Hung/Desktop/new_cool_input.txt"
f = open(filename)
k=open("C:/Users/Zac Hung/Desktop/another_cool_input.txt","w")
lines = f.readlines()
for line in lines:
    k.write(line.split("&")[1]+" "+line.split("&")[4])
