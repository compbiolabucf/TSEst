import os,sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LoadDataset(Dataset):
    def __init__(self, file_path, seq_len, feature_num, model_type):
        super(LoadDataset, self).__init__()
        self.file_path = file_path
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.model_type = model_type


class LoadValTestDataset(LoadDataset):
    """Load val test set"""
    def __init__(self, file_path, set_name, seq_len, feature_num, model_type):
        super(LoadValTestDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf[set_name]['X'][:]
            self.X_hat = hf[set_name]['X_hat'][:]
            self.missing_mask = hf[set_name]['missing_mask'][:]
            self.indicating_mask = hf[set_name]['indicating_mask'][:]
            self.SNP = hf[set_name]['SNP'][:]

        self.X = np.nan_to_num(self.X)
        self.X_hat = np.nan_to_num(self.X_hat)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
       
        sample = (
            torch.tensor(idx),
            torch.from_numpy(self.X_hat[idx].astype('float32')),
            torch.from_numpy(self.missing_mask[idx].astype('float32')),
            torch.from_numpy(self.X[idx].astype('float32')),
            torch.from_numpy(self.indicating_mask[idx].astype('float32')),
            torch.from_numpy(self.SNP[idx].astype('float32')),
        )

        return sample


class LoadTrainDataset(LoadDataset):
    """Load train set"""

    def __init__(self, file_path, seq_len, feature_num, model_type, masked_imputation_task, MIT_RND,artificial_missing_rate):
        super(LoadTrainDataset, self).__init__(file_path, seq_len, feature_num, model_type)
        self.masked_imputation_task = masked_imputation_task
        self.MIT_RND = MIT_RND
        if masked_imputation_task:
            self.artificial_missing_rate = artificial_missing_rate
            assert 0 < self.artificial_missing_rate < 1, 'artificial_missing_rate should be greater than 0 and less than 1'

        with h5py.File(self.file_path, 'r') as hf:  # read data from h5 file
            self.X = hf['train']['X'][:]
            self.SNP = hf['train']['SNP'][:]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X[idx]

        if self.MIT_RND is False:
            """For TS missing pattern imputation"""
            X_mat = X[:,0]  #only first feature is enough to know availability 
            """Finding indices to mask"""
            indices = np.where(~np.isnan(X_mat))[0].tolist()
            indices = np.random.choice(indices, round(len(indices) * self.artificial_missing_rate))
            X_hat = np.copy(X)

            indicating_mask = np.zeros(len(X_mat)).astype(int)
            indicating_mask[indices] = 1
            #indicating_mask = indicating_mask.reshape([sample_num, seq_len])
            indicating_mask = np.expand_dims(indicating_mask,1)
            indicating_mask = np.tile(indicating_mask,(1,self.feature_num))
            X_hat[indicating_mask==1] = np.nan  # X_hat contains artificial missing values
            """All masks including original missing"""
            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            """Only artificial missing"""
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)

        else:
            """For RND missing pattern imputation"""
            X = X.reshape(-1)
            indices = np.where(~np.isnan(X))[0].tolist()
            indices = np.random.choice(indices, round(len(indices) * self.artificial_missing_rate))
            X_hat = np.copy(X)
            X_hat[indices] = np.nan  # mask values selected by indices

            missing_mask = (~np.isnan(X_hat)).astype(np.float32)
            indicating_mask = ((~np.isnan(X)) ^ (~np.isnan(X_hat))).astype(np.float32)

            X = X.reshape(self.seq_len, self.feature_num)
            X_hat = X_hat.reshape(self.seq_len, self.feature_num)
            missing_mask = missing_mask.reshape(self.seq_len, self.feature_num)
            indicating_mask = indicating_mask.reshape(self.seq_len, self.feature_num)
           
        X = np.nan_to_num(X)
        X_hat = np.nan_to_num(X_hat)

        sample = (
            torch.tensor(idx),
            torch.from_numpy(X_hat.astype('float32')),
            torch.from_numpy(missing_mask.astype('float32')),
            torch.from_numpy(X.astype('float32')),
            torch.from_numpy(indicating_mask.astype('float32')),
            torch.from_numpy(self.SNP[idx].astype('float32')),   
        )
    
        return sample


class UnifiedDataLoader:
    def __init__(self, dataset_path, seq_len, feature_num, model_type, batch_size=1024, num_workers=4,
                 masked_imputation_task=False, MIT_RND = False, artificial_missing_rate=0.2):

        self.dataset_path = os.path.join(dataset_path, 'datasets.h5')
        self.seq_len = seq_len
        self.feature_num = feature_num
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.model_type = model_type
        self.masked_imputation_task = masked_imputation_task
        self.MIT_RND = MIT_RND
        self.artificial_missing_rate = artificial_missing_rate

        self.train_dataset, self.train_loader, self.train_set_size = None, None, None
        self.val_dataset, self.val_loader, self.val_set_size = None, None, None
        self.test_dataset, self.test_loader, self.test_set_size = None, None, None

    def get_train_val_dataloader(self):
        self.train_dataset = LoadTrainDataset(self.dataset_path, self.seq_len, self.feature_num, self.model_type,
                                              self.masked_imputation_task,self.MIT_RND,self.artificial_missing_rate)
        self.val_dataset = LoadValTestDataset(self.dataset_path, 'val', self.seq_len, self.feature_num, self.model_type)
        self.train_set_size = self.train_dataset.__len__()
        self.val_set_size = self.val_dataset.__len__()
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.val_loader = DataLoader(self.val_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.train_loader, self.val_loader

    def get_test_dataloader(self):
        self.test_dataset = LoadValTestDataset(self.dataset_path, 'test', self.seq_len, self.feature_num,
                                               self.model_type)
        self.test_set_size = self.test_dataset.__len__()
        self.test_loader = DataLoader(self.test_dataset, self.batch_size, shuffle=True, num_workers=self.num_workers)
        return self.test_loader
