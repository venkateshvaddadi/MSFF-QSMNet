<html>
<head>
</head>
<body>



<h1>MSFF-QSMNet: An Efficient Multi-Scale Feature Fusion Network for QSM Reconstruction</h1>

<h2>MSFF-QSMNet Architecture</h2>

<img src='images/MSFF-QSMNet.png'>

<h2>Sample QSM reconstruction on QSM-2016 challenge data</h2>  
  
<img src="data/qsm_2016_recon_challenge/output/spinet_qsm_output_image.png" alt="spinet-QSM architecture" width=100% height=100%>


<h2>MSFF-QSMNet reconstruction on complete training data</h2>
<p>
QSM maps reconstructed on complete training data were shown in the first row, and residual error maps in the second row.
</p>

  <img src="images/spinet_qsm_on_complete_training_data.png" alt="spinet-QSM architecture" width=100% height=100%>
# How to run the code
First, ensure that PyTorch 1.10 or higher version is installed and working with GPU. Second, just clone or download this reporsitory. The SpiNet_QSM_demo_run.py file should run without any changes in the code. 

We can run from the command prompt: **`python SpiNet_QSM_demo_run.py`**.
We can check the quality of the Spinet-QSM output by running the **`metrics_evaluation.m`**. It calculates the **`SSIM, pSNR, RMSE, HFEN`**.
# Dependencies
<li> Python  </li>  
<li> PyTorch 1.10 </li>
<li> MATLAB R2022b </li>
  
# Files description
**`savedModels:`** This directory contain's the learned PyTorch 1.10 model parameters. 

**`SpiNet_QSM_demo_run.py:`** It to read the model and run on the given demo input data from Data folder.

**`dw_WideResnet.py:`** This file contains the 3D-WideResNet(Residual learning CNN model) code for the denoiser($D_{w}$). 

**`loss.py:`** This file contains the code for the loss function. the $l_1$-norm of the **voxel-wise difference** (L1 loss term) and **gradient-wise difference** (edge loss) was utilized in a weighted manner.

**`utils.py:`** This file contains the code for many supporting functions for the previous Python code files.
# Contact
Dr. Phaneendra K. Yalavarthy

Assoc.Prof, CDS, IISc Bangalore, email : yalavarthy@iisc.ac.in

Vaddadi Venkatesh

(PhD) CDS, MIG, IISc Bangalore, email : venkateshvad@iisc.ac.in
</body>
</html>
