#Graph convolutional neural network with baynesian inference on smiles dataset to classify mutagenicity

#Hirohara, M., Saito, Y., Koda, Y. et al. Convolutional neural network based on SMILES representation of compounds for detecting chemical #motif. BMC Bioinformatics 19, 526 (2018). https://doi.org/10.1186/s12859-018-2523-5

Format the dataset in exact format(with even the same spacing)
and run in the terminal in pycharm with the following command to train the model
python trainer-challenge.py --gpu=-1 -i ./TOX21  -p NR-AR

Run the following line on the test set to get the baynesian statistics:
python evaluate-challenge.py --gpu=-1 -m ./TOX21/ -d ./TOX21 -p NR-AR

