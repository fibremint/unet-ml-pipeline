# Components

* **check-deployable**: Compares the metric between previously deployed model and another one that trained on the pipeline. If trained model’s performance (accuracy) is better, it would be designated as to be deployed.
* **check-preprocess-proceed**: The newly added data have to be pre-processed. This module checks whether data were added to data storage or not, and outputs its checked result.
* **data-preprocess**: Generates image patches and corresponding mask image with a size of 256x256
* **deploy**: Simply shows the information of deployment configuration.
* **evaluate** : Evaluates the trained model.
* **train** : Train a model.
* **update-deploy-config**: Updates deployment configuration with the  designated model.
