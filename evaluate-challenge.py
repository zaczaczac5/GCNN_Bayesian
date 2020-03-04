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


# -------------------------------------------------------------
def main():
    # 引数管理
    parser = argparse.ArgumentParser(description='SMILES CNN fingerprint')
    parser.add_argument('--batchsize', '-b', type=int, default=32,
                        help='Number of moleculars in each mini-batch. Default = 32')
    parser.add_argument('--epoch', '-e', type=int, default=40,
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
    #file = args.data + '/' + args.protein + '_test.smiles'
    #file = args.data + '/' + args.protein + '_wrong_classification.smiles'
    file = args.data + '/' + args.protein + '_new.smiles'
    print('Loading smiles: ', file)
    smi = Chem.SmilesMolSupplier(file, delimiter='\t', titleLine=False)
    mols = [mol for mol in smi if mol is not None]

    F_list, T_list = [], []
    for mol in mols:
        #modify here
        if len(Chem.MolToSmiles(mol, kekuleSmiles=True, isomericSmiles=True)) > 400:
            print("too long mol was ignored")
        else:
            F_list.append(mol_to_feature(mol, -1, 400))
            T_list.append(mol.GetProp('_Name'))
    Mf.random_list(F_list)
    Mf.random_list(T_list)



    # -------------------------------
    # reset memory
    #del mol, mols, data_f, F_list, T_list
    gc.collect()

    # -------------------------------
    print('Evaluater is  running...')

    # -------------------------------
    # Set up a neural network to evaluate
    model = Mm.CNN(400, lensize, args.k1, args.s1, args.f1, args.k2, args.s2, args.k3, args.s3, args.f3,
                   args.k4, args.s4, args.n_hid, args.n_out)
    model.compute_accuracy = False
    #model.to_gpu(args.gpu)
    f = open(args.model + '/' + args.protein + '/evaluation_epoch.csv', 'w')

    # -------------------------------
    print("epoch", "TP", "FN", "FP", "TN", "Loss", "Accuracy", "MCC", "Specificity", "Precision", "Recall",
          "F1", sep="\t")
    f.write("epoch,TP,FN,FP,TN,Loss,Accuracy,MCC,Specificity,Precision,Recall,F1\n")
    data_t = np.asarray(T_list, dtype=np.int32).reshape(-1, 1)
    data_f = np.asarray(F_list, dtype=np.float32).reshape(-1, 1, 400, lensize)
    #print(data_t.shape, data_f.shape)
    borders = [len(data_t) * i // 30 for i in range(30 + 1)]

    data_f = np.array(data_f)
    data_t = np.array(data_t)
    for epoch in range(args.frequency, args.epoch + 1, args.frequency):

        pred_score, loss = [], []


        serializers.load_npz(args.model + '/' + args.protein + '/model_snapshot_' + str(epoch), model)

        for i in range(30):

            x_gpu = data_f[borders[i]:borders[i + 1]]
            y_gpu = data_t[borders[i]:borders[i + 1]]
            pred_tmp_gpu, sr = model.predict(Variable(x_gpu))
            pred_tmp_gpu = F.sigmoid(pred_tmp_gpu)
            pred_tmp = pred_tmp_gpu.data
            loss_tmp = model(Variable(x_gpu), Variable(y_gpu)).data
            pred_score.extend(pred_tmp.reshape(-1).tolist())
            loss.append(loss_tmp.tolist())
        loss = np.mean(loss)
        pred_score = np.array(pred_score).reshape(-1, 1)
        pred = 1 * (pred_score >= 0.5)

        count_TP = np.sum(np.logical_and(data_t == pred, pred == 1) * 1).astype(float)
        count_FP = np.sum(np.logical_and(data_t != pred, pred == 1) * 1).astype(float)
        count_FN = np.sum(np.logical_and(data_t != pred, pred == 0) * 1).astype(float)
        count_TN = np.sum(np.logical_and(data_t == pred, pred == 0) * 1).astype(float)

        Accuracy = (count_TP + count_TN) / (count_TP + count_FP + count_FN + count_TN)
        Sepecificity = count_TN / (count_TN + count_FP)
        Precision = count_TP / (count_TP + count_FP)
        Recall = count_TP / (count_TP + count_FN)
        Fmeasure = (2 * count_TP)/ (2*count_TP+count_FP+count_FN)
        import math
        MCC = (count_TP*count_TN-count_FP*count_FN)/math.sqrt(abs((count_TN
                                                                          +count_FP)
                                                                         *(count_TN+count_FN)
                                                                         *(count_TP+count_FP)
                                                                         *(count_TP+count_FN)))


        print(epoch, count_TP, count_FN, count_FP, count_TN, loss, Accuracy, MCC, Sepecificity, Precision,
              Recall, Fmeasure, sep="\t")
        text = '{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}\n'.format(
            epoch, count_TP, count_FN, count_FP, count_TN, loss, Accuracy, MCC, Sepecificity, Precision, Recall,
            Fmeasure)
        f.write(text)

    f.close()


# -------------------------------
if __name__ == '__main__':
    main()
