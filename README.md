# cupsnbottles-data-analysis
Provides data analysis tools for ISY project "Multimodal object tracking on a mobile robot".
Some code was already provided by Christian Limberg (https://ieee-dataport.org/open-access/cupsnbottles) and altered according to the needs of the project.  
It aims to meaningfully analyze image data especially with respect to ambiguous samples and overlapping objects.

---

## Data Format
As can be seen from **dataset03/**, data should contain an **image/** folder with images of format .png or .jpeg.  
The filenames should be increasing integer values, ideally starting at 0.

Correspondingly, a **properties.csv** should be provided of following format, with column **index** also being the filename of the respective sample:

| index  | object_class  | ambiguous | overlap  |
| ------ |:-------------:| :--------:| :------: |
| 0      | waterbottle   | 1         | 0        |
| 1      | coke          | 0         | 1        |

Running `python data_preprocessing.py 'path_to_dataset'` will extract features from the images using a trained VGG16 network that can be classified.

---

## Grid Search

Running `python grid_search.py` will perform a grid search on the dataset specified in **config.ini** and following classifiers:  
* Nearest Neighbors
* Linear SVM
* RBF SVM
* Gaussian Process
* Decision Tree
* Random Forest
* Neural Net
* Naive Bayes
* QDA
* GLVQ

The most suitable parameters will be saved for each classifier to later be used in the evaluation step in **/classifiers_best_params**.   
CSV files containing information on the grid search process are saved there as well.

---

## Evaluation / Analysis

All details for the evaluation can be customized in the **config.ini** file, including:
* path to dataset
* classifier
* normal_evaluation (bool)
* or a manual evaluation split where different categories of data can be split to ones needs (vanilla, ambiguous, overlap)

If not specified otherwise, the evaluation process will automatically load the best parameters for the classifier as found in the grid search and start training. Analysis involves plots to highlight the classification performance (e.g. an overview, confusion matrices and scatterplots with depicted samples). The plots use a t-SNE embedding of the high-dimensional data and are saved to **/plots**. Additionally, a CSV file found in **/evaluation** will save meaningful information on misclassified samples.

The process is started by running `python evaluation.py`.

---

## Plot Examples
### Image of an overview
![Image of Overview](https://github.com/JacquelineKonkol/cupsnbottles-data-analysis/blob/j-dev/plots/RBF_SVM.png)
### Image of a scatterplot
![Image of Scatterplot](https://github.com/JacquelineKonkol/cupsnbottles-data-analysis/blob/j-dev/plots/linear_svmlowest_confidence.png)
### Image of a confusion matrix
![Image of Confusion Matrix](https://github.com/JacquelineKonkol/cupsnbottles-data-analysis/blob/j-dev/plots/conf_matrix_linear_svmnorm_cupsnbottles.png)
