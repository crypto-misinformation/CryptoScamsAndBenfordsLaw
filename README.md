# CryptoScamsAndBenfordsLaw

### Pulling Amberdata

You can pull the data from Amberdata using the get_amberdata_v2.py function. Note you may need to create subfolder names if you change save file names.

You will also need an amberdata key. If you cannot get an amberdata key in your country, another option is to pull data using the etherscan API (you can get a free lisence on their website). I do not have scripts to request etherscan data but as long as the csv files have the following columns it should work with the rest of the code.

- "blockNumber", "from", "to", "gasLimit", "value"

### Creating feature matrices

This can be done with the gen_feature_matrices file. The way that script is set up is that it will read csv files create created by the get_amberdata_v2 script. You can also choose to not save the csv file and perform searches through an enormous dataframe but I found that typically that df was large enough to crash the terminal session.

If you do decide not to use the csv files, you will need to change the code in the gen_feature_matrices to not look for the csv files.

### Training the Models

Everything for the training and analysis of the 5 models we tested are in the respective files. The are all scripts except for the lightGBM model which is in a jupyter notebook (this is bc it was more complicated to debug and get the data set up right lol). If you do want to seperate the training and evaulation processes for each model, you'll need to break up the functions.


### Citation
Also, if you decide to use this work for your research, please cite the associated paper:

> **_Citation:_** 
J. Gridley and O. Seneviratne, "Significant Digits: Using Large-Scale Blockchain Data to Predict Fraudulent Addresses," 2022 IEEE International Conference on Big Data (Big Data), Osaka, Japan, 2022, pp. 903-910, doi: 10.1109/BigData55660.2022.10020971.

