import glob
import numpy as np
from alive_progress import alive_bar
from sklearn import metrics
import matplotlib.pyplot as plt
import time

class Part1:
    def execute(self):
        for name in glob.glob("detection_files/*.txt"):
            print("========================\File: ", name)
            tp_count = 0
            fp_count = 0
            tn_count = 0
            fn_count = 0
            for seuil in {0.1, 0.5, 0.7}:
                data = np.loadtxt(name, dtype=np.float32, delimiter=' ')

                for elem in data:
                    if elem[0] >= seuil:
                        if elem[1] == 1:
                            tp_count += 1
                        else:
                            fp_count += 1
                    else:
                        if elem[1] != 1:
                            tn_count += 1
                        else:
                            fn_count += 1

                print("-----------------------------\nSeuil: ", seuil)
                print("TP: ", tp_count)
                print("FP: ", fp_count)
                print("TN: ", tn_count)
                print("FN: ", fn_count)

                recall = tp_count / (tp_count + fn_count)
                precision = tp_count / (tp_count + fp_count)

                print("Recall: ", recall)
                print("Precision: ", precision)


class Part2:
    data = np.array([]).reshape(0, 2)

    tp_count = 0
    fp_count = 0
    tn_count = 0
    fn_count = 0

    result = []

    def execute(self):
        self.load_all_data(self.get_files())
        seuils = self.find_all_seuil()

        with alive_bar(7) as bar:
            for name in glob.glob("detection_files/*.txt"):
                data = np.loadtxt(name, dtype=float, delimiter=' ')

                for seuil in seuils:
                    if seuil > 1:
                        break

                    self.tp_count = 0
                    self.fp_count = 0
                    self.tn_count = 0
                    self.fn_count = 0

                    self.compute_matrix(data, seuil)

                    recall = self.tp_count / (self.tp_count + self.fn_count)
                    try:
                        precision = self.tp_count / (self.tp_count + self.fp_count)
                    except:
                        # Parfois le seuil est trop haut et on a pas de tp
                        pass

                    self.result_dict = {}

                    self.result_dict['file'] = name
                    self.result_dict['seuil'] = seuil
                    self.result_dict['recall'] = recall
                    self.result_dict['precision'] = precision

                    self.result.append(self.result_dict)
                
                bar()

        self.result = np.array(self.result)

        precision_array = self.get_precision(self.result)
        recall_array = self.get_recall(self.result)

        sorted_index = np.argsort(precision_array)
        precision_array = precision_array[sorted_index]
        recall_array = recall_array[sorted_index]

        auc = metrics.auc(precision_array, recall_array)
        print("AUPR: ", auc)

        for file in glob.glob("detection_files/*.txt"):
            precision_array = self.get_precision_by_file(self.result, file)
            recall_array = self.get_recall_by_file(self.result, file)
            plt.plot(recall_array, precision_array, label=file.split('/')[1].split('.')[0] + " AUC: " + str(auc))
            plt.legend()
        plt.show()
            

    def compute_matrix(self, dataset, threshold):
        sup_th = np.where(dataset[:, 0] >= threshold)
        inf_th = np.where(dataset[:, 0] < threshold)
        self.tp_count = len(np.where(dataset[sup_th[0], 1] == 1)[0])
        self.fp_count = len(sup_th[0]) - self.tp_count
        self.tn_count = len(np.where(dataset[inf_th[0], 1] == 0)[0])
        self.fn_count = len(inf_th[0]) - self.tn_count

    def get_precision(self, _dict):
        values = [item.get('precision') for item in _dict]
        values_array = np.array(values)
        return values_array
    
    def get_recall(self, _dict):
        values = [item.get('recall') for item in _dict]
        values_array = np.array(values)
        return values_array
    
    def get_precision_by_file(self, values, file):
        out = []
        for elem in values:
            if elem.get('file') == file:
                out.append(elem.get('precision'))
        return np.array(out)
    
    def get_recall_by_file(self, values, file):
        out = []
        for elem in values:
            if elem.get('file') == file:
                out.append(elem.get('recall'))
        return np.array(out)          

    def get_files(self):
        return glob.glob("detection_files/*.txt")
    
    def load_all_data(self, files):
        for file in files:
            temp = np.genfromtxt(file, dtype=np.float32, delimiter=' ')
            self.data = np.concatenate((self.data, temp), axis=0)

    def find_all_seuil(self):
        return np.unique(self.data[:, 0])

if __name__ == "__main__":
    # _1 = Part1()
    # _1.execute()

    _2 = Part2()
    _2.execute()
            