# Test and reproduction of the "OT improves cell-cell similarity inference in single-cell omics data" paper

These data and Python script will allow you to run properly our reproduction of the '[Optimal Transport improves cell-cell similarity inference in single-cell omics data](https://academic.oup.com/bioinformatics/article/38/8/2169/6528312)' paper (Huizing et al., 2022) computations. 
Optimal Transport is there applied as a similarity metric in between single-cell omics data.

## Jupyter Notebook
The notebook is available on Google Colab : https://colab.research.google.com/drive/1rsGUv4wT_lJfTWyCju3ohCC0fo-g_26F?usp=sharing

The first part, which demonstrate the general computations on one reduced dataset (keeping only the 1,000 most varying features instead of 10,000), is meant to be run on a GPU. You may then want to use the one available on Google Colab for free.

The second part, which iterates on all datasets keeping this time the  10,000 most varying features, cannot be run on Google Colab without running out of memory. We ran it on a GPU servor and saved the results for each dataset in a json file, that we provide in this GitHub repository and that is meant to be loaded on the notebook.

You can still look at the steps we used to produce this data on the notebook.
The results are shown at the end as barplots.

### test_functions
Those are the implementation of the whole functions used in the first part of Jupyter Notebook, copied from the authors' [original Notebook](https://colab.research.google.com/github/ComputationalSystemsBiology/OT-scOmics/blob/main/docs/source/vignettes/OT_scOmics.ipynb).

### data_results.json
The analysis on 10,000 rows cannot be run on Google collab without running out of memory. We thus ran it on a GPU servor and saved the results as a json file, meant to be loaded on the notebook.

**How the data is stored :**

The data is stored in dictionnaries, one for each score. The different items correspond to the scores obtained with the different datasets. The keys are the names of the files related to the data.


For the *S and C scores*, the scores for each dataset are stored in a np.ndarray which shape is (2,1).


*  The first row corresponds to the Pearson correlation scores. The second one to the OT scores.

For the ARI, NMI, AMI scores and the number of clusters (n_clus), the scores for each dataset are stored in a np.ndarray which shape is (2,3).


*   The first row corresponds to the Pearson correlation scores. The second one to the OT scores.
*   The first column corresponds to the supervised Hierarchical (HS) cluctering results. The second column corresponds to the unsupervised Hierarchical (HU) cluctering results. The third one corresponds to the Spectral cluctering (S) results.


## Supplementary Text 2
We also wanted to try to reproduce the analysis mentionned in the [Supplementary material text 2](https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bioinformatics/38/8/10.1093_bioinformatics_btac084/1/btac084_supplementary_data.zip?Expires=1659429360&Signature=nMn7Eju9SgLcv0wpX~JzvvFCSFokofXn95wYKUFEVS0yE2HjawQRjz9p~w1BgrO54ni5mmYxXMN1zMkWyeeZglQz-~m9vvJDhm1TUr17kLvxqPKII2es0XdLEWOPcQh64EFvUxRVF3WTG5dCBoOYT-WH8oPt1z-t1DIA52BTLgVcuvcXoG4xmMuiJK4dj1dGSRIpbDm1nLRTx9~ZGSSR8K8zWuY-nQZN47219VU3wNGLXcVZ0RasGFa9C3Yxa3udTVWYNmckyWcp9GRymluBeiqBpq-67HFKEzORM8j3s7VGOrZBM7u9c9~dKKnDBNoA9PFhJD-H0bPCjnB~uVI6FQ__&Key-Pair-Id=APKAIE5G5CRDK6RD3PGA), untitled *'Analysis of the transport plans between cells with similar activated transcriptional programs'*.

We based our analysis on the same dataset (the Li Tumor (Li et al., 2017) dataset) that we provide on this GitHub for more convenience.

Another notebook for this specific analysis is available here : https://colab.research.google.com/drive/1cK3AFOOS82chwneU1zotIgIR_z9JE_Qm?usp=sharing


---
**Reference :**

Huizing,G.-J. et al. (2022) Optimal transport improves cell–cell similarity inference in single-cell omics data. Bioinformatics **38**, 2169–2177

Liu,L. et al. (2019) Deconvolution of single-cell multi-omics layers reveals regulatory heterogeneity. Nat Commun, **10**, 470
