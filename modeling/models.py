import numpy as np

from modeling.layers import *
from modeling.utils import masked_mae_cal
import seaborn as sns
import matplotlib.pyplot as plt   

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        """ Sinusoid position encoding table """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()


class TransformerEncoder(nn.Module):
    def __init__(self, n_groups, n_group_inner_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
                 **kwargs):
        super().__init__()
        self.n_groups = n_groups
        self.d_time = d_time
        self.n_group_inner_layers = n_group_inner_layers
        self.input_with_mask = kwargs['input_with_mask']
        actual_d_feature = d_feature * 2 if self.input_with_mask else d_feature
        self.param_sharing_strategy = kwargs['param_sharing_strategy']
        self.MIT = kwargs['MIT']
        self.device = kwargs['device']
        self.d_feature = d_feature
        
        self.COMP_2_layers = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
            for _ in range(n_groups)])
     
        self.COMP_12_layers = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
            for _ in range(n_groups)])
        
        self.COMP_112_SA_layers = nn.ModuleList([
            EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, dropout, **kwargs)
            for _ in range(n_groups)])
       
        #self.COMP_111_layers = FeedForward_Teddy(50,d_feature)
        #self.COMP_111_layers = TEDDY_LSTM(14,d_feature,self.device)
        self.COMP_111_layers = FeedForward_Camel(27,d_feature)
        
        self.embedding_112 = nn.Linear(d_feature, d_model)
        self.embedding_12 = nn.Linear(actual_d_feature, d_model)
        
        self.embedding = nn.Linear(actual_d_feature, d_model)
        self.position_enc = PositionalEncoding(d_model, n_position=d_time)
        self.dropout = nn.Dropout(p=dropout)
        self.reduce_dim_11 = nn.Linear(d_model, d_feature)
        self.reduce_dim_12 = nn.Linear(d_model, d_feature)
        self.reduce_dim_2 = nn.Linear(d_model, d_feature)
        self.criterion = nn.MSELoss()
        #self.reduce_dim_temp = nn.Linear(2*d_feature, d_feature)
        
    def impute(self, inputs,stage,idx=None):
        X, X_holdout, masks, SNP = inputs['X'], inputs['X_holdout'], inputs['missing_mask'], inputs['SNP']

        # feed forward / LSTM for COMP 11        
        #SNP = SNP.repeat(1,1,1).permute(1, 0, 2).repeat(1,self.d_time,1)
        X_tilde_111 = self.COMP_111_layers(SNP)
        X_tilde_111 = X_tilde_111.repeat(1,1,1).permute(1, 0, 2).repeat(1,self.d_time,1)
        
        # SA for COMP 11
        input_X = self.embedding_112(X_tilde_111)
        enc_output = self.dropout(self.position_enc(input_X))

        for encoder_layer in self.COMP_112_SA_layers:
            enc_output, attn_weights_COMP11 = encoder_layer(enc_output)
        X_tilde_11 = self.reduce_dim_11(enc_output)
        attn_weights_COMP11 = attn_weights_COMP11.detach()

        #X =X_tilde_11

        # COMP 12
        input_X = torch.cat([X, masks], dim=2) if self.input_with_mask else X
        input_X = self.embedding_12(input_X)
        
        enc_output = self.dropout(self.position_enc(input_X))

        for encoder_layer in self.COMP_12_layers:
            enc_output, attn_weights_COMP12 = encoder_layer(enc_output)
        X_tilde_12 = self.reduce_dim_12(enc_output)
        attn_weights_COMP12 = attn_weights_COMP12.detach()
        
        STEP1_loss = self.criterion(X_tilde_11,X_tilde_12)
        STEP1_loss += masked_mae_cal(X_tilde_11, X, masks)*0.000     

        attn_weights_COMP11 = torch.transpose(attn_weights_COMP11, 1, 3)
        attn_weights_COMP11 = attn_weights_COMP11.mean(dim=3)
        attn_weights_COMP11 = torch.transpose(attn_weights_COMP11, 1, 2)

        attn_weights_COMP12 = torch.transpose(attn_weights_COMP12, 1, 3)
        attn_weights_COMP12 = attn_weights_COMP12.mean(dim=3)
        attn_weights_COMP12 = torch.transpose(attn_weights_COMP12, 1, 2)

        n_sample = X.size()[0]
        NEW_ATT1 = torch.empty((n_sample,self.d_time),device = self.device)
        NEW_ATT2 = torch.empty((n_sample,self.d_time),device = self.device)

        for sample_idx in range(n_sample):
            if stage == 'train':
                corr1 = torch.diagonal(torch.corrcoef(torch.cat([X_holdout.detach()[sample_idx,:,:], X_tilde_11.detach()[sample_idx,:,:]],0))[self.d_time:,:self.d_time], 0)
                corr2 = torch.diagonal(torch.corrcoef(torch.cat([X_holdout.detach()[sample_idx,:,:], X_tilde_12.detach()[sample_idx,:,:]],0))[self.d_time:,:self.d_time], 0)
            else:
                corr1 = torch.diagonal(torch.corrcoef(torch.cat([X.detach()[sample_idx,:,:], X_tilde_11.detach()[sample_idx,:,:]],0))[self.d_time:,:self.d_time], 0)
                corr2 = torch.diagonal(torch.corrcoef(torch.cat([X.detach()[sample_idx,:,:], X_tilde_12.detach()[sample_idx,:,:]],0))[self.d_time:,:self.d_time], 0)


            corr1 = torch.where(corr1>=0,corr1,torch.exp(-3*corr1)/5 ) #
            corr2 = torch.where(corr2>=0,corr2,torch.exp(-3*corr2)/5 )

            SUM = torch.add(corr1,corr2)
            corr1 = corr1/SUM
            corr2 = corr2/SUM

            corr1 = corr1.repeat(self.d_time,1).permute(1,0)
            corr2 = corr2.repeat(self.d_time,1).permute(1,0)

            NEW_ATT1[sample_idx,:] = torch.nanmean((attn_weights_COMP11[sample_idx,:,:]*corr1),0)
            NEW_ATT2[sample_idx,:] = torch.nanmean((attn_weights_COMP12[sample_idx,:,:]*corr2),0)

        SUM = torch.add(NEW_ATT1,NEW_ATT2)
        NEW_ATT1 = NEW_ATT1/SUM
        NEW_ATT2 = NEW_ATT2/SUM

        NEW_ATT1 = torch.nan_to_num(NEW_ATT1, nan=0)
        NEW_ATT2 = torch.nan_to_num(NEW_ATT2, nan=0)
        
        #temp=torch.concat((NEW_ATT1,NEW_ATT2),1).cpu().numpy()
        #np.savetxt('test'+str(idx)+'.csv',temp,fmt='%s',delimiter=',')
        
        NEW_ATT1 = torch.unsqueeze(NEW_ATT1,2).repeat(1,1,self.d_feature)
        NEW_ATT2 = torch.unsqueeze(NEW_ATT2,2).repeat(1,1,self.d_feature)

        #temp = torch.cat([X_tilde_11, X_tilde_12], dim=2)
        #X_tilde_C = self.reduce_dim_temp(temp)
        X_tilde_C = NEW_ATT1*X_tilde_11 + NEW_ATT2*X_tilde_12
        #X_tilde_C = X_tilde_11 + X_tilde_12
        
        #if stage!='train':
        #    X_tilde_C = X_tilde_11
        #X_prime = masks * X + (1 - masks) * X_tilde_11
        X_prime = masks * X + (1 - masks) * X_tilde_C
        
        
        # COMP 2
        input_X = torch.cat([X_prime, masks], dim=2) if self.input_with_mask else X_prime #X_prime
        input_X = self.embedding(input_X)

        enc_output = self.dropout(self.position_enc(input_X))

        for encoder_layer in self.COMP_2_layers:
            enc_output, _ = encoder_layer(enc_output)
        
        learned_presentation = self.reduce_dim_2(enc_output)
        imputed_data = masks * X + (1 - masks) * learned_presentation  # replace non-missing part with original data
        
        return imputed_data, learned_presentation, STEP1_loss

    def forward(self, inputs, stage,idx=None):
        X, masks = inputs['X'], inputs['missing_mask']
        imputed_data, learned_presentation, STEP1_loss = self.impute(inputs,stage,idx=idx)
        reconstruction_MAE = masked_mae_cal(learned_presentation, X, masks)
        if (self.MIT or stage == 'val') and stage != 'test':
            # have to cal imputation loss in the val stage; no need to cal imputation loss here in the test stage
            imputation_MAE = masked_mae_cal(learned_presentation, inputs['X_holdout'], inputs['indicating_mask'])
           
        else:
            imputation_MAE = torch.tensor(0.0)

        return {'imputed_data': imputed_data,
                'reconstruction_loss': reconstruction_MAE, 'imputation_loss': imputation_MAE,
            'reconstruction_MAE': reconstruction_MAE, 'imputation_MAE': imputation_MAE, 'STEP1_loss': STEP1_loss}

