# MLOps Pipeline Components

This components are the modules that used in the pipeline that conducts continuous training which based on argo workflow. The workflow proceeds each of the steps by pulling the container image that required on a specific task.

## Continuous Deployment with GitHub Actions

The tasks that defined in a workflow proceeds its own job by pulling the container image that published on the Container Registry.

This deployment process like building a module, writing some specific tag to image and push to the remote repository could be fulfilled manually on a local development machine. However, deployment can be processed automatically on the this component repository with GitHub Actions. 

The deployment workflow do build and publish only for updated components. It looks at component path and detects changes by comparing to the previous commits. This deployment process would be applied to the each of updated modules.

## References

https://github.com/zizhaozhang/nmi-wsi-diagnosis

