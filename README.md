## Road Lane Detection
This is the public repository for Comprehensive Design (Machine Intelligence/Digital Media), Software Engineering School, Tongji University.

Several files are not uploaded to this public repo, including trained models, model implementations. **So the software IN THIS REPO cannot be launched or cannot function normally**



### Functions Implemented

- Models to predict lane marks
  - Multi-task model for lane prediction and arrow segmentation is included
- Cross-platform interactive interface for detection



### Supported Platforms

- Web Browser
  - Require ECMAScript 7 Support (or Babel Compiler)
- Android or Android-based OS (like MIUI & FlymeOS)
  - Require Android Webview Version >= 79
- Windows 10+
- Unix-like System / MacOS
  - Require Rebuild Electron Project



### How To Run 

#### Prerequisites

**Environment Prerequsites**

Basic frameworks & enviroment configurations. Items marked in bold text are necessary.

| Purpose/Item    | Requirement                                                  |
| --------------- | ------------------------------------------------------------ |
| **Python**      | Version >=3.9                                                |
| **Web Browser** | Supports ECMAScript 7 (without Babel.js)                     |
| **Node.js**     |                                                              |
| Android Webview | (Necessary if you uses the android distribution)<br/>Version >= 79 |
| Java            | Version=8                                                    |
| Android SDK     | Version=30                                                   |



**Computing Prerequsites**

| Purpose/Item                  | Version Requirement |
| ----------------------------- | ------------------- |
| CUDA (Training)               | >= 10.2             |
| CUDA (CPU Inference)          | >= 10.2             |
| CUDA (CUDA Inference)         | >= 11.0             |
| CUDA (TensorRT Inference)     | >= 11.0             |
| TensorRT (Training)           | Unnecessary         |
| TensorRT (CPU Inference)      | Unnecessary         |
| TensorRT (CUDA Inference)     | Unnecessary         |
| TensorRT (TensorRT Inference) | >=8.2               |
| **Python**                    | **>=3.9**           |



**Runtime Libraries Required：**

Libraries below should be added to the `PATH` environmental variable for Windows. And these can be installed via `apt-get` in Linux.

- libjpeg-turbo (Official Page:https://libjpeg-turbo.org/)



**Python Packages Required:**

Versions of packages are significantly crucial. Mismatched versions will cause the unexpected behaviours of the application. Check package versions via `pip3 list` or `python -m pip list`.

| Package Name                  | Version Requirement    |
| ----------------------------- | ---------------------- |
| OpenCV (opencv-python)        | **>=  4.5.3**          |
| PyTorch (torch)               | **>= 1.8 (>=cu102)**   |
| PyTorch Vision (torchvision)  | **>= 0.9.0 (>=cu102)** |
| ONNXRuntime (onnxruntime-gpu) | <b>\~1.10.0</b>        |
| ONNXRuntime (onnxruntime)     | <b>\~1.10.0</b>        |
| albumentations                | **>= 1.0.0 < 1.1.0**   |
| segmentation-models-pytorch   | >= 0.2.0               |
| torchsummary                  | >= 1.5.0               |
| PyYAML                        | >= 5.4.0               |
| torchmetrics                  | **>= 0.7.3**           |
| tqdm                          | >= 4.62.0              |
| matplotlib                    | >= 3.4.0               |
| jpeg4py                       | >=0.1.0                |
| NumPy                         | Depend on dependencies |
| Pandas                        | >= 1.3.2               |
| scikit-learn                  | >= 0.24.0              |
| scipy                         | >= 1.7.0               |
| flask                         |                        |
| flask-socketio                |                        |



#### Steps

**Install Packages**: Check whether prerequisites above are met.

```
pip install <package>
conda install <package>
```

**To Train The Model**：Modify `global_config_server_side.yaml` if you want to train your model at the server side. Otherwise, modify `global_config_client_side.yaml`. Make sure that the paths to the dataset are correct.

Then

```
python train.py
```

You have to select to whether to train the model on your server or not via an input prompt. This dialog window will only appear once.

 **To Deploy The Model**: Just follow the instruction below

```
python deploy.py
```

**To Run Frontend**

```
npm install
npm run dev
```

### 

### Acknowledgements

**Top Code & Resource References**

| Project                                                      | Type              | License        | Source     |
| ------------------------------------------------------------ | ----------------- | -------------- | ---------- |
| **SCNN**<br/>XingangPan                                      | Model             | MIT            | GitHub     |
| **SCNN_Pytorch** <br/>harryhan618                            | Model             | MIT            | GitHub     |
| **YOLOP** <br/>HUST Visual Learning Team (HUSTVL)            | Model             | MIT            | GitHub     |
| **Ultra Fast Lane Detection**<br/>cfzd                       | Model             | MIT            | GitHub     |
| **Novecento Sans** (Normal)<br/>**Novecento Sans** (Wide Bold)<br/>Jan Tonellato | UI/UX - Font      | -              | Adobe Font |
| **Source Han Sans / Noto Sans Han**<br/>Adobe / Google       | UI/UX - Font      | SIL Open Font  | GitHub     |
| **Bender** <br/>Jovanny Lemonad                              | UI/UX - Font      | -              | -          |
| **Geometos** <br/>Deepak Dogra                               | UI/UX - Font      | Non-commercial | DAFont     |
| **Source Han Serif <br/>Noto Sans  Han Serif**<br/>Adobe / Google | UI/UX - Font      | SIL Open Font  | GitHub     |
| **Naive UI**<br/>TuSimple Inc.                               | UI/UX - Component | MIT            | GitHub     |
| **XIcons**<br/>07akioni                                      | UI/UX - Icons     | -              | GitHub     |
| **Proton-Engine-Example**<br/>drawcall                       | UI/UX - SFX       | MIT            | GitHub     |
| **OpenCV.js**<br/>OpenCV                                     | Graphics          | Apache         | OpenCV     |
| **Cordova**                                                  | Android Migration | Apache         | NPM        |
| **Vue.js** (v3)                                              | Frontend          | MIT            | NPM        |
| **Flask**                                                    | Backend           | BSD-3          | PIP        |
| **PyTorch**                                                  | AI Training       | -              | PIP        |
| **ONNXRuntime**                                              | AI Inference      | MIT            | PIP        |
| **OpenCV** (Python)                                          | Image Processing  | Apache         | PIP        |
| **Albumentation**                                            | Image Processing  | MIT            | PIP        |
| **Segmentation-Models-Pytorch**                              | Backbone          | MIT            | PIP        |



**Additional Statements**

License for XIcons coincides with its source repositories including [`fluentui-system-icons`](https://github.com/microsoft/fluentui-system-icons), [`ionicons`](https://github.com/ionic-team/ionicons), [`ant-design-icons`](https://github.com/ant-design/ant-design-icons), [`material-design-icons`](https://github.com/google/material-design-icons),



### Collaborators

This repository contains codes form following collaborators:

https://github.com/HugePotatoMonster/

https://github.com/mhy-666

