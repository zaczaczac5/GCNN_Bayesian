#!/usr/bin/env python
# coding:utf-8

import argparse
import gc

import numpy as np
#import cupy as cp
import pandas as pd

from rdkit import Chem
# from rdkit.Chem import AllChem
# from rdkit.Chem import rdchem
from feature import *
import SCFPfunctions as Mf
import SCFPmodel as Mm
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
from sklearn import metrics

import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, serializers
from chainer import Link, Chain, ChainList
from chainer.datasets import tuple_dataset
from chainer.training import extensions

# -------------------------------------------------------------
# featurevector size
atomInfo = 21
structInfo = 21
lensize = atomInfo + structInfo
#np.random.seed(0)
def softmax(x):
    r=np.exp(x - np.max(x))
    return r/r.sum(axis=0)
def np_sigmoid(x):
    return 1./(1.+np.exp(-x))
npa = np.array
def softmax1(w, t = 1.0):
    e = np.exp(npa(w) / t)
    dist = e / np.sum(e)
    return dist


def softmax2(x):
    """
    Compute softmax values for each sets of scores in x.

    Rows are scores for each class.
    Columns are predictions (samples).
    """
    scoreMatExp = np.exp(np.asarray(x))
    return scoreMatExp / scoreMatExp.sum(0)
# -------------------------------------------------------------
def main():
    # 引数管理
    parser = argparse.ArgumentParser(description='SMILES CNN fingerprint')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of moleculars in each mini-batch. Default = 32')
    parser.add_argument('--epoch', '-e', type=int, default=1,
                        help='Number of max iteration to evaluate. Default = 20')
    parser.add_argument('--gpu', '-g', type=int, default=-1, help='GPU ID (negative value indicates CPU). Default = -1')
    parser.add_argument('--frequency', '-f', type=int, default=1, help='Epoch frequency for evaluation. Default = 1')
    parser.add_argument('--model', '-m', help='Directory to Model to evaluate')
    parser.add_argument('--data', '-d', required=True, help='Input Smiles Dataset')
    parser.add_argument('--protein', '-p', required=True, help='Name of protein (subdataset)')
    parser.add_argument('--k1', type=int, default=11, help='window-size of first convolution layer. Default = 11')
    parser.add_argument('--s1', type=int, default=1, help='stride-step of first convolution layer. Default = 1')
    parser.add_argument('--f1', type=int, default=128, help='No. of filters of first convolution layer. Default = 128')
    parser.add_argument('--k2', type=int, default=5, help='window-size of first pooling layer. Default = 5')
    parser.add_argument('--s2', type=int, default=1, help='stride-step of first max-pooling layer. Default = 1')
    parser.add_argument('--k3', type=int, default=11, help='window-size of second convolution layer. Default = 11')
    parser.add_argument('--s3', type=int, default=1, help='stride-step of second convolution layer. Default = 1')
    parser.add_argument('--f3', type=int, default=64, help='No. of filters of second convolution layer. Default = 64')
    parser.add_argument('--k4', type=int, default=5, help='window-size of second pooling layer. Default = 5')
    parser.add_argument('--s4', type=int, default=1, help='stride-step of second pooling layer. Default = 1')
    parser.add_argument('--n_out', type=int, default=1, help='No. of output perceptron (class). Default = 1')
    parser.add_argument('--n_hid', type=int, default=96, help='No. of hidden perceptron. Default = 96')

    args = parser.parse_args()

    # -------------------------------
    print('Making Test Dataset...')
    #file = args.data + '/' + args.protein + '_w.smiles'
    file = args.data + '/' + args.protein + '_randWrong.smiles'
    #file = args.data + '/' + args.protein + '_wholeTest.smiles'
    #file = args.data + '/' + args.protein + '_japanval.smiles'
    #file = args.data + '/' + args.protein + '_wrong_classification.smiles'
    #file = args.data + '/' + args.protein + '_new.smiles'
    #file = args.data + '/' + args.protein + '_w.smiles'
    #file = args.data + '/' + args.protein + '_simpleTest.smiles'
    print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file, delimiter='\t', titleLine=False)
    mols = [mol for mol in smi if mol is not None]

    F_list, T_list = [], []
    euclidean_d=[]
    #chec=list(range(1,100,2))
    for mol in mols:
        tmp=mol_to_feature(mol, -1, 200)
        #modify here
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > 200:
            print("too long mol was ignored",Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True))
            print(len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)))
        else:
            F_list.append(tmp)
            euclidean_d.append(np.sum(np.square(tmp)))
            T_list.append(mol.GetProp('_Name'))
    #Mf.random_list((chec))
    #Mf.random_list(F_list)
    #Mf.random_list(T_list)
    #Mf.random_list((chec))
    #print(chec)
    #print(np.asarray(T_list))



    # -------------------------------
    # reset memory
    #del mol, mols, data_f, F_list, T_list
    gc.collect()

    # -------------------------------
    print('Evaluater is  running...')

    # -------------------------------
    # Set up a neural network to evaluate
    model = Mm.CNN(200, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3,
                   args.k4, args.s4, args.n_hid, args.n_out)
    model.compute_accuracy = False
    #model.to_gpu(args.gpu)
    f = open(args.model + '/' + args.protein + '/evaluation_epoch.csv', 'w')

    # -------------------------------
    print("epoch", "TP", "FN", "FP", "TN", "Uncertainty", "Accuracy", "MCC", "Specificity", "Precision", "Sensitivity",
          "F1","ale_uncertainty","epi_uncertainty", sep="\t")
    f.write("epoch,TP,FN,FP,TN,Uncertainty,Accuracy,MCC,Specificity,Precision,Sensitivity,F1,ale_uncertainty,epi_uncertainty\n")
    data_t = np.asarray(T_list, dtype=np.int32).reshape(-1, 1)
    data_f = np.asarray(F_list, dtype=np.float32).reshape(-1, 1, 200, lensize)
    #print(data_t.shape, data_f.shape)
    borders = [len(data_t) * i // 100 for i in range(100 + 1)]

    data_f = np.array(data_f)
    data_t = np.array(data_t)
    for epoch in range(args.frequency, args.epoch + 1, args.frequency):

        pred_score, loss = [], []
        lamp,phone,big=[],[],[]





        serializers.load_npz(args.model + '/' + args.protein + '/model_snapshot_' + str(100), model)
        #monte carlo sampling, predict for many 100s times to predict different outputs to use it for uncertainty
        for i in range(100):
            yes = []
            for j in range(5):
                x_gpu = data_f[borders[i]:borders[i + 1]]
                y_gpu = data_t[borders[i]:borders[i + 1]]
                pred_tmp_gpu, sr = model.predict(Variable(x_gpu))
                #print(pred_tmp_gpu)
                #pred_tmp_gpu = F.sigmoid(pred_tmp_gpu)
                pred_tmp = pred_tmp_gpu.data
                loss_tmp = model(Variable(x_gpu), Variable(y_gpu)).data
                yes.append(pred_tmp.flatten())
            king=np_sigmoid(np.asarray(yes))
            pred_tmp=(np.mean(king,axis=0))
            pred_score.extend(pred_tmp.reshape(-1).tolist())
            loss.append(loss_tmp.tolist())
            #print(np.asarray(yes).shape)
            pred_output = np_sigmoid(np.asarray(yes))

            #pred_output = (yes)
            #pred_output=[[0.9,0.99,0.98,0.5,0.55,0.001,0.1,0.2],[0.1,0.99,0.98,0.5,0.55,0.001,0.1,0.2]]

            pred_output=np.asarray(pred_output)
            #print(np.mean(pred_output,axis=0))
            # #take the mean of the array of the calculation of different outputs generated from loops above
            ale_unc = np.mean(pred_output * (1.0 - pred_output),axis=0)
            #print(np.asarray(ale_unc).shape)
            # #calculate epi uncertainty
            epi_unc = np.mean(pred_output ** 2,axis=0) - np.mean(pred_output,axis=0) ** 2
            # #total uncertainty
            pred_output_sum = ale_unc + epi_unc
            lamp.append(ale_unc)
            phone.append(epi_unc)
            big.append(pred_output_sum)

            #print('fingerprint1',np.sum(np.square((model.fingerprint(x_gpu))[0,:,:,:])))
            #print('fingerprint2', np.sum(np.square((model.fingerprint(x_gpu))[1, :, :, :])))

        #print('fingerprintfinal', np.sum(np.square((model.fingerprint(x_gpu))[2, :, :, :])))

        ale=np.mean(np.concatenate(lamp))
        epi=np.mean(np.concatenate(phone))
        tot=np.mean(np.concatenate(big))


        print('coolooo',np.asarray(pred_score).shape)
        #pred_score=np.mean(No,axis=0)
        pred_score = np.array(pred_score).reshape(-1, 1)
        pred = 1 * (pred_score >= 0.5)

        count_TP = np.sum(np.logical_and(data_t == pred, pred == 1) * 1).astype(float)
        count_FP = np.sum(np.logical_and(data_t != pred, pred == 1) * 1).astype(float)
        count_FN = np.sum(np.logical_and(data_t != pred, pred == 0) * 1).astype(float)
        count_TN = np.sum(np.logical_and(data_t == pred, pred == 0) * 1).astype(float)

        Accuracy = (count_TP + count_TN) / (count_TP + count_FP + count_FN + count_TN)
        Specificity = count_TN / (count_TN + count_FP)
        Precision = count_TP / (count_TP + count_FP)
        #sensitivity
        Recall = count_TP / (count_TP + count_FN)
        #F1
        Fmeasure = (2 * count_TP)/ (2*count_TP+count_FP+count_FN)
        import math
        MCC = (count_TP*count_TN-count_FP*count_FN)/math.sqrt(abs((count_TN
                                                                          +count_FP)
                                                                         *(count_TN+count_FN)
                                                                         *(count_TP+count_FP)
                                                                         *(count_TP+count_FN)))


        print(epoch, count_TP, count_FN, count_FP, count_TN,tot, Accuracy, MCC, Specificity, Precision,
              Recall, Fmeasure,ale,epi, sep="\t")
        text = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n'.format(
            epoch, count_TP, count_FN, count_FP, count_TN, tot, Accuracy, MCC, Specificity, Precision, Recall,
            Fmeasure,ale,epi)
        f.write(text)

        print('ale',(np.concatenate(lamp, axis=0)))
        print('epi', np.concatenate(phone,axis=0))
        print('tot', np.concatenate(big,axis=0))
        #print('score',(pred_score))
        print('pred_output', (pred))
        print('euclidean distance:',(euclidean_d))



    f.close()


# -------------------------------
if __name__ == '__main__':
    main()
