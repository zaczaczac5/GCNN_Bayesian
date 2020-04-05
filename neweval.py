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
    #file = args.data + '/' + args.protein + '_wholeTest.smiles'
    #file = args.data + '/' + args.protein + '_wrong_classification.smiles'
    #file = args.data + '/' + args.protein + '_new.smiles'
    #file = args.data + '/' + args.protein + '_simple.smiles'
    #file = args.data + '/' + args.protein + '_simpleTest.smiles'
    file = args.data + '/' + args.protein + '_wrongex.smiles'
    print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file, delimiter='\t', titleLine=False)
    mols = [mol for mol in smi if mol is not None]

    F_list, T_list = [], []
    #grape=list(range(1, 202))
    #grapej = list(range(0, 402,2))
    for mol in mols:
        #modify here
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > 550:
            print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol, -1, 550))
            T_list.append(mol.GetProp('_Name'))
    #Mf.random_list(F_list)
    #Mf.random_list(T_list)
    #Mf.random_list(grape)
    #Mf.random_list(grapej)
    #print(grape)
    #print(grapej)
    #print(len(grape))
    #print(len(grapej))


    # -------------------------------
    # reset memory
    #del mol, mols, data_f, F_list, T_list
    gc.collect()

    # -------------------------------
    print('Evaluater is  running...')

    # -------------------------------
    # Set up a neural network to evaluate
    model = Mm.CNN(550, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3,
                   args.k4, args.s4, args.n_hid, args.n_out)
    model.compute_accuracy = False
    #model.to_gpu(args.gpu)
    f = open(args.model + '/' + args.protein + '/evaluation_epoch.csv', 'w')

    # -------------------------------
    print("epoch", "TP", "FN", "FP", "TN", "Uncertainty", "Accuracy", "MCC", "Specificity", "Precision", "Sensitivity",
          "F1","ale_uncertainty","epi_uncertainty", sep="\t")
    f.write("epoch,TP,FN,FP,TN,Uncertainty,Accuracy,MCC,Specificity,Precision,Sensitivity,F1,ale_uncertainty,epi_uncertainty\n")
    data_t = np.asarray(T_list, dtype=np.int32).reshape(-1, 1)
    data_f = np.asarray(F_list, dtype=np.float32).reshape(-1, 1, 550, lensize)
    #print(data_t)
    borders = [len(data_t) * i // 100 for i in range(100 + 1)]

    data_f = np.array(data_f)
    data_t = np.array(data_t)
    #print(data_t)
    for epoch in range(args.frequency, args.epoch + 1, args.frequency):

        pred_score, loss = [], []
        cool,fat=[],[]
        pizza,yogurt=[],[]

        serializers.load_npz(args.model + '/' + args.protein + '/model_snapshot_' + '3', model)

        #monte carlo sampling, predict for many 100s times to predict different outputs to use it for uncertainty
        for i in range(100):

            x_gpu = data_f[borders[i]:borders[i + 1]]
            y_gpu = data_t[borders[i]:borders[i + 1]]
            pred_tmp_gpu, sr = model.predict(Variable(x_gpu))
            why=pred_tmp_gpu.data
            why=1 * (why >= 0.5)
            pizza.append(x_gpu)
            #print((why))
            tmp = softmax(why)
            m = np.mean(tmp)
            ep = np.mean(np.square(tmp - m))
            dg = np.diag(tmp)
            prod = np.square(tmp)
            ai = np.mean(dg - prod)
            cool.append(ai)
            fat.append(ep)

            pred_tmp_gpu = F.sigmoid(pred_tmp_gpu)
            pred_tmp = pred_tmp_gpu.data
            loss_tmp = model(Variable(x_gpu), Variable(y_gpu)).data
            pred_score.extend(pred_tmp.reshape(-1).tolist())
            loss.append(loss_tmp.tolist())
        #create prediction array
        #print((pizza))
        pred_output=np_sigmoid(np.asarray(pred_score))
        # #take the mean of the array of the calculation of different outputs generated from loops above
        ale_unc = np.mean(cool)
        epi_unc = np.mean(fat)

        #ale_unc = np.mean(pred_output * (1.0 - pred_output))
        # #calculate epi uncertainty
        #epi_unc = np.mean(pred_output ** 2) - np.mean(pred_output) ** 2
        # #total uncertainty
        pred_output= ale_unc + epi_unc
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


        print(epoch, count_TP, count_FN, count_FP, count_TN, pred_output, Accuracy, MCC, Specificity, Precision,
              Recall, Fmeasure,ale_unc,epi_unc, sep="\t")
        text = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11},{12},{13}\n'.format(
            epoch, count_TP, count_FN, count_FP, count_TN, pred_output, Accuracy, MCC, Specificity, Precision, Recall,
            Fmeasure,ale_unc,epi_unc)
        f.write(text)

    f.close()
    print('ale',cool)
    print('epi',fat)
    print(cool+fat)
# -------------------------------
if __name__ == '__main__':
    main()
