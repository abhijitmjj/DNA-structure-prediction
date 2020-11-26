
from collections import defaultdict
import math
import pandas as pd 
import numpy as np
import seaborn as sns
import statistics
import sklearn

from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support
from sklearn.metrics import average_precision_score
from matplotlib import pyplot
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import os
from pathlib import Path


class Call_Plot():
    
    def __init__(self, sklearn_model=True, model_name="SVM", repeated_k_fold=False, DIR=Path(os.getcwd())):
        plt.close()
        self.DIR = DIR
    
        self.model_name = model_name
        self.fig, self.ax = plt.subplots()
        self.ax.plot([0,1], [0,1], linestyle='--', label='Random choice')
        self.ax.set_xlabel('False Positive Rate', fontsize=12)
        self.ax.set_ylabel('True Positive Rate', fontsize=12)
        
        self.fig2, self.ax2 = plt.subplots()
        self.ax2.set_xlabel('Recall', fontsize=12)
        self.ax2.set_ylabel('Precision', fontsize=12)

        self.tprs = []
        self.aucs = []
        self.mean_fpr = np.linspace(0, 1, 100)
        self.no_skill = []
        self.sklearn_model = sklearn_model
        self.results = defaultdict(list)
        self.repeated_k_fold = repeated_k_fold
        
        
    def Plot(self, data: dict, model, idx):
        if self.sklearn_model:
            y_pred_val = model.predict_proba(data["X_val"])[:,1]
        else:
            y_pred_val = model.predict(data["X_val"])
        
        #Precision-Recall
        precision, recall, thresholds = precision_recall_curve(data["y_val"], y_pred_val)
        no_skill = len(data["y_val"][data["y_val"]==1]) / len(data["y_val"])
        self.no_skill.append(no_skill)
        avg_pr = average_precision_score(data["y_val"], y_pred_val)
        auc_pr = sklearn.metrics.auc(recall, precision)
        if self.repeated_k_fold:
            self.ax2.plot(recall, precision, marker='.', label=f'Run {(idx)//5+1} Test Fold{(idx)%5+1}: AUC PR={auc_pr:.2f}')
        else:
            self.ax2.plot(recall, precision, marker='.', label=f'Test Fold{(idx)+1}: AUC PR={auc_pr:.2f}')
        
        # convert to f score
        fscore = (2 * precision * recall) / (precision + recall)
        # locate the index of the largest f score
        ix_pr = np.argmax(fscore)
        self.ax2.scatter(recall[ix_pr], precision[ix_pr], marker='o', color='black')
        
        Accuracy = sklearn.metrics.accuracy_score(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))
        target_names = ['B-DNA', 'A-DNA']
        print(classification_report(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0), target_names=target_names))
        F1 = sklearn.metrics.f1_score(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))
        MCC = sklearn.metrics.matthews_corrcoef(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))
        cohen_kappa_score = sklearn.metrics.cohen_kappa_score(data["y_val"], np.where(y_pred_val > thresholds[ix_pr], 1, 0))

        
        #ROC-AUC
        fpr, tpr, thresholds_auc = roc_curve(data["y_val"], y_pred_val)
        # calculate the g-mean for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        # locate the index of the largest g-mean
        ix = np.argmax(gmeans)
        if self.repeated_k_fold:
            self.ax.plot(fpr, tpr, marker='.',
                     label=f'Run {(idx)//5+1} Test Fold{(idx)%5+1}: AUC={sklearn.metrics.auc(fpr, tpr):.2f}')
        else:
            self.ax.plot(fpr, tpr, marker='.',
                     label=f'Test Fold{(idx)+1}: AUC={sklearn.metrics.auc(fpr, tpr):.2f}')
        self.ax.scatter(fpr[ix], tpr[ix], marker='o', color='black')
        # axis labels
        self.ax.legend(loc="lower left")
        # Mean plot
        interp_tpr = np.interp(self.mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        self.tprs.append(interp_tpr)
        self.aucs.append(gmeans[ix])
        

        print("Average PR: ", avg_pr )
        print("AUC PR: ", auc_pr)
        print('Best Threshold_f-score=%f, F-Score=%.3f' % (thresholds[ix_pr], fscore[ix_pr]))
        print("AUC: ", sklearn.metrics.auc(fpr, tpr))
        print('Best Threshold_ROC=%f, G-Mean_ROC=%.3f' % (thresholds_auc[ix], gmeans[ix]))
        print("Accuracy: ", Accuracy )
        print("F1: ", F1 )
        print("MCC: ", MCC )
        self.results["Average PR"].append(avg_pr)
        self.results["AUC PR"].append(auc_pr)
        self.results["ROC AUC"].append(sklearn.metrics.auc(fpr, tpr))
        self.results["Accuracy"].append(Accuracy)
        self.results["F1"].append(F1)
        self.results["MCC"].append(MCC)
        self.results["cohen_kappa_score"].append(cohen_kappa_score)

    
    def post_Plot(self):
        from sklearn.metrics import auc
        mean_tpr = np.mean(self.tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(self.mean_fpr, mean_tpr)
        std_auc = np.std(self.aucs)
        self.ax.plot(self.mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

        std_tpr = np.std(self.tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        self.ax.fill_between(self.mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                        label=r'$\pm$ 1 std. dev.')
        self.ax.legend(loc=(0.45, .05),fontsize='medium')
        self.fig.savefig(self.DIR/"data"/"results"/f"{self.model_name}_AUC_ROC.png", dpi=600)
        no_skill = np.mean(np.array(self.no_skill))
        self.ax2.plot([0,1], [no_skill,no_skill], linestyle='--', label="Random")
        self.ax2.legend(loc=(0.050, .08),fontsize='medium')
        self.fig2.savefig(self.DIR/"data"/"results"/f"{self.model_name}_AUC_PR.png", dpi=600)
        