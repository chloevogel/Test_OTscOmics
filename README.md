# Test of the "OT improves cell-cell similarity inference in single-cell omics data" paper

These data and Python script will allow you to run properly the test on the '[Optimal Transport improves cell-cell similarity inference in single-cell omics data](https://academic.oup.com/bioinformatics/article/38/8/2169/6528312)' paper. 
Optimal Transport is there applied as a similarity metric in between single-cell omics data.

## Jupyter Notebook
The notebook is available on Google Colab : https://colab.research.google.com/drive/1ExY9QUKArbspCRYQIvPssSb8fCfkDW8b?usp=sharing

The first part, which demonstrate the general computations on one reduced dataset (keeping only the 1,000 most varying features instead of 10,000), is meant to be run on a GPU. You may then want to use the one available on Google Colab for free.

The second part, which iterates on all datasets keeping this time the  10,000 most varying features, cannot be run on Google Colab without running out of memory. We ran it on a GPU servor and save the results for each dataset in a json file, that we provide in this GitHub repository and that is meant to be loaded on the notebook.

You can still look at the steps we used to produce these datas on the notebook.
The results are shown at the end as barplots.
