# TSEst
TSEst is a multi-modal time series imputation model that can use an additional modality (another cross-sectional or time-series data) to impute missing values in a time series data. This multi-modal approach shows improved performance over uni-modal imputation models. 

### **Framework**
<img src="https://github.com/compbiolabucf/TSEst/blob/main/Fig-1.png" width="450" height="450">

## Environment
Environment can be created using the command **conda env create -f my_conda_env.yml**. [my_conda_env.yml](https://github.com/compbiolabucf/TSEst/blob/main/my_conda_env.yml) is provided in the repository. 

## Quick start guide
Download a sample data (Daymet) from this [link](https://knightsucfedu39751-my.sharepoint.com/:f:/g/personal/t_ahmed_knights_ucf_edu/EqcCFQeTVg5HgGUuA7SwQmUBZZB6cVJNXXO3CT_OAWr30w?e=uw6AaT) into the parent directory. Run **python3 run_models.py --config_path configs/Camel_Transformer_best_rnd.ini** to train the model. Modify the values of <model_saving_dir> and 




