import torch, pickle
import numpy as np, random
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
def load(dataset_name):
    """Load dataset from file.
    Args:
        dataset_name (str): Key to retrieve the file path from a predefined dictionary.
    Returns:
        dict: Loaded data from the pickle file.
    """
    filenames = {
        "DSA_original":               "datasets/original/Daily_and_Sports_Activities_Data_Set.pickle", 
        "PAMAP_original":             "datasets/original/PAMAP2.pickle", 
        "WISDM-phone_original":       "datasets/original/WISDM_accel_phone.pickle", 
        "WISDM-watch_original":       "datasets/original/WISDM_accel_watch.pickle", 
        "REALDISP-ideal_original":    "datasets/original/REALDISP_ideal.pickle", 
        "REALDISP-mutual_original":   "datasets/original/REALDISP_mutual.pickle", 
        "REALDISP-self_original":     "datasets/original/REALDISP_self.pickle", 
        "VPA_original":               "datasets/original/Vicon_Physical_Action.pickle", 
        "ADL_original":               "datasets/original/ADL_HMP_Dataset.pickle", 
        
        "DSA_processed":              "datasets/processed/Daily_and_Sports_Activities_Data_Set.pickle", 
        "PAMAP_processed":            "datasets/processed/PAMAP2.pickle", 
        "WISDM-phone_processed":      "datasets/processed/WISDM_accel_phone.pickle", 
        "WISDM-watch_processed":      "datasets/processed/WISDM_accel_watch.pickle", 
        "REALDISP-ideal_processed":   "datasets/processed/REALDISP_ideal.pickle", 
        "REALDISP-mutual_processed":  "datasets/processed/REALDISP_mutual.pickle", 
        "REALDISP-self_processed":    "datasets/processed/REALDISP_self.pickle", 
        "VPA_processed":              "datasets/processed/Vicon_Physical_Action.pickle", 
        "ADL_processed":              "datasets/processed/ADL_HMP_Dataset.pickle", 
    }
    
    with open(filenames[dataset_name], 'rb') as handle:
        return pickle.load(handle)
        
# -------------------------------------------------------------------
class DataLoader:
    """Handles loading and preparing datasets for training and testing."""
    
    def __init__(self, dataset_name):
        """Initialize DataLoader with datasets.
        Args:
            dataset_name (str): Name of the dataset to load both original and processed data.
        """
        self.data_ori = load(dataset_name + "_original")
        self.data_ext = load(dataset_name + "_processed")
        
        persons = list(self.data_ext.keys())
        self.persons = sorted(persons) 
        
        classes = list(set([ c for j in self.data_ext.keys() for c in self.data_ext[j].keys() ]))
        self.classes = sorted(classes) 
    
    # -------------------------------------------------------------------
    def sample_dataset(self, n_classes, scale=True):
        # Data of a randomly selected person
        j = random.choice(self.persons)
        dico_ori = self.data_ori[j]
        dico_ext = self.data_ext[j]
        
        # Randomly select n_classes classes
        classes_j =  list( set(self.classes).intersection( set(dico_ext.keys()) ) ) # in self.classes and person j have them
        classes = random.sample(classes_j, min(n_classes, len(classes_j)))  
        
        # Randomly map each selected class to a label in {0, ..., n_classes-1}
        labels = random.sample(range(n_classes), n_classes)
        label_map = dict(zip(classes, labels))
        
        X_ori, X_ext, y = [], [], []
        for c in classes:
            Xc_ori = random.sample(list(dico_ori[c]), len(dico_ori[c]))
            Xc_ext = random.sample(list(dico_ext[c]), len(dico_ext[c]))
            
            X_ori += Xc_ori
            X_ext += Xc_ext
            y += [label_map[c] for _ in Xc_ext]
        
        X_ori = np.array(X_ori)
        X_ext = np.array(X_ext)
        y = np.array(y)
        
        if scale:
            X_ext = StandardScaler(with_mean=True, with_std=True).fit_transform(X_ext)

        # Transform these lists to appropriate tensors and return them
        X_ori = torch.from_numpy(X_ori).float()
        X_ext = torch.from_numpy(X_ext).float()
        y = torch.from_numpy(y).long()
        
        return X_ori, X_ext, y
