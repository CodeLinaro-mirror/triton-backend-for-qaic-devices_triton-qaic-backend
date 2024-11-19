# QAIC Backend

The Triton backend for [Qualcomm Cloud AI 100](https://www.qualcomm.com/products/technology/processors/cloud-artificial-intelligence/cloud-ai-100#Overview). You can learn more about Triton backends in the [backend repo](https://github.com/triton-inference-server/backend).
Ask questions or report problems in the main Triton [issues page](https://github.com/triton-inference-server/server/issues).

The QAIC backend is designed to deploy inference execution on compiled pre-trained models in QAic Program Container (QPC) format. See [here](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Inference-Workflow/) for instruction to convert a model to QPC format. The backend is implemented using QAIC C++ API.

## Build the QAIC backend

### Pre-requisites

* Download and Install [Qualcomm Cloud AI 100 Platform SDK](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/#platform-sdk) on the host bare metal OS or VM with AIC card(s).

### Option 1. Build docker container from Qualcomm Cloud AI 100 SDK

* Download and unzip [Qualcomm Cloud AI 100 Apps SDK](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/#apps-sdk).
* Run the docker build script with the specified configuration.

  ```bash
  $ cd </path/to/apps-sdk>/common/tools/docker-build

  $ python3 build_image.py --image_name qaic-triton --tag <sdk-version> --log_level 2 --user_specification_file </path/to/apps-sdk>/common/tools/docker-build/sample_user_specs/user_image_spec_triton_model_repo.json --apps-sdk /apps/sdk/path.zip --platform-sdk /platform/sdk/path.zip
  ```

  The build command above generates incremental docker image for triton with qaic backend support in local docker repository.

  ```bash
    $ docker image ls
    REPOSITORY   TAG                 IMAGE ID       CREATED         SIZE
    qaic-triton  <sdk-version>       a0968cf3711b   3 days ago      27.3GB
  ```

### Option 2. Build backend library from source in Default Triton Container

* You can follow steps described in the [Building With Docker](https://github.com/triton-inference-server/server/blob/main/docs/customization_guide/build.md#building-with-docker) guide and use the [build.py](https://github.com/triton-inference-server/server/blob/main/build.py) script with desired customization to build a triton container.
* Clone triton-qaic-backend.
* Launch the tritonserver_buildbase container with qaic_backend volume.

  ```bash
  $ docker run -it -v /path/to/triton-qaic-backend:/qaic_backend tritonserver_buildbase:latest bash
  ```

* Install Qualcomm Cloud AI 100 SDK.

  ```bash
  $ apt-get update && apt-get install pciutils sudo
  ```

  Download and Install [Qualcomm Cloud AI 100 Apps and Platform SDK](https://quic.github.io/cloud-ai-sdk-pages/latest/Getting-Started/Installation/Cloud-AI-SDK/Cloud-AI-SDK/) within the container.

* Within "qaic_backend" directory, execute the following steps.
  ``` bash
  $ mkdir build
  $ cd build
  $ cmake [-DCMAKE_BUILD_TYPE=Debug] -DCMAKE_INSTALL_PREFIX:PATH=`pwd`/install ..
  $ make install
  ```

  The following required Triton repositories will be pulled and used in the build. By default the "main" branch/tag will be used for each repo but the listed CMake argument can be used to override.
  * triton-inference-server/backend: -DTRITON_BACKEND_REPO_TAG=[tag]
  * triton-inference-server/core: -DTRITON_CORE_REPO_TAG=[tag]
  * triton-inference-server/common: -DTRITON_COMMON_REPO_TAG=[tag]

* Deploy qaic backend at triton installation.
  * Launch tritonserver with qaic_backend volume

    ```bash
    $ docker run -it -v /path/to/triton-qaic-backend:/qaic_backend --privileged --shm-size=4g --ipc=host --net=host tritonserver:latest bash
    ```

  * Follow steps mentioned above to Install Qualcomm Cloud AI 100 SDK within the container.
  * Copy backend library.

    ```bash
    $ cd /opt/tritonserver/backends
    $ mkdir qaic
    $ cp /qaic_backend/build/libtriton_qaic.so /opt/tritonserver/backends/qaic/
    ```

## Using the QAIC backend

### Parameters

For qaic backend configuration, set the following parameters in model config file -

* `backend`: qaic backend is identified as "qaic".

   ```pbtxt
   backend: "qaic"
   ```

* `instance_group`: Set the model instance group kind as "KIND_MODEL".

   ```pbtxt
   instance_group [
    {
      count: 2
      kind: KIND_MODEL
    }
   ]
   ```

#### AIC100 specific parameters

User-provided key-value pairs which Triton will pass to backend runtime environment as variables and can be used in processing logic of backend.

### Supported features

* Model Ensemble
* Dynamic Batching
* Auto-Pick Devices
* Auto-Complete Model Configuration
