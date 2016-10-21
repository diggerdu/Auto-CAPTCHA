# Auto-CAPTCHA
available for the website of SHU
- R-CNN 
combine Lenet-5 and Bi-directional RNN, Thus far accuracy is 97.6% in validation set

##GUIDE
1. `git clone git@github.com:diggerdu/Auto-CAPTCHA.git`
2. Download Model Checkpoint from https://eyun.baidu.com/s/3bpmlsVP
3. `cp model.ckpt Auto-CAPTCHA/R-CNN/checkpoint/`
4. `python Auto-CAPTCHA/R-CNN/predict-Bi-R-CNN.py <your-image-file-path>`
##TODO LIST
construct a CTC model to further increase the performance 
