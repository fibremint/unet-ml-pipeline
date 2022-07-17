# MLOps Pipeline Components

This components are the modules that used in the pipeline that conducts continuous training which based on argo workflow. The workflow proceeds each of the steps by pulling the container image that required on a specific task.

## Components

* **check-deployable**: Compares the metric between previously deployed model and another one that trained on the pipeline. If trained model’s performance (accuracy) is better, it would be designated as to be deployed.
* **check-preprocess-proceed**: The newly added data have to be pre-processed. This module checks whether data were added to data storage or not, and outputs its checked result.
* **data-preprocess**: Generates image patches and corresponding mask image with a size of 256x256
* **deploy**: Simply shows the information of deployment configuration.
* **evaluate** : Evaluates the trained model.
* **train** : Train a model.
* **update-deploy-config**: Updates deployment configuration with the  designated model.

## Continuous Deployment with GitHub Actions

The tasks that defined in a workflow proceeds its own job by pulling the container image that published on the Container Registry.

This deployment process like building a module, writing some specific tag to image and push to the remote repository could be fulfilled manually on a local development machine. However, deployment can be processed automatically on the this component repository with GitHub Actions. 

The deployment workflow do build and publish only for updated components. It looks at component path and detects changes by comparing to the previous commits. This deployment process would be applied to the each of updated modules.

## References
* Original Codes: https://github.com/zizhaozhang/nmi-wsi-diagnosis
* Conditional actions execution based on files modification: https://github.com/dorny/paths-filter
* Define environment within an actions: https://github.com/actions/starter-workflows/issues/68
* Build and push the docker image to GCR: http://acaird.github.io/computers/2020/02/11/github-google-container-cloud-run
* GCR access control: https://cloud.google.com/container-registry/docs/access-control
* GCR authentication to docker: https://cloud.google.com/container-registry/docs/advanced-authentication#gcloud-helper

### Troubleshoot
* https://github.com/dorny/paths-filter/issues?q=is%3Aissue+is%3Aclosed+not+a+git+repository+128
