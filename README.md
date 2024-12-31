# A Study on Context Length and Efficient Transformers for Biomedical Image Analysis

This repository provides code for the experiments reported in our ML4H 2024 Paper, A Study on Context Length and Efficient Transformers for Biomedical Image Analysis. 

## Setup

#### Environment 
We recommend using a virtual enviroment. We have tested this code using Python 3.11 and CUDA 12.4. 

Start by installing pytorch 2.1.1 and torchvision 0.16.1. Installation instructions for these packages can vary by user setup, see the official Pytorch documentation to get the correct install files. The installation commands that work for us are:
```
pip install torch==2.1.1 torchvision==0.16.1 --index-url https://download.pytorch.org/whl/cu118
```

You will also need to install the causal-conv1d 1.1.1 and mamba-ssm 1.2.0 packages provided by https://github.com/state-spaces/mamba. Installation instructions for these packages can vary by user setup, see the linked github repo for more details. The installation commands that work for us are:
```
pip install causal-conv1d==1.1.1 mamba-ssm==1.2.0.post1
```

Finally, clone this repo and install the remaining Python dependencies by running the following commands. 

```
  git clone https://github.com/sarahmhooper/lc_imaging.git
  cd lc_imaging
  pip install -r requirements.txt
```





#### Data preparation

Data sources and preprocessing are described in our paper. The codebase can be used with these public datasets or with custom datasets, as long as data is formatted according to the following structure:

```
├── dataset_name
│   ├── subject_1
│   │   ├── subject_1_input.npy
│   │   ├── subject_1_output.npy (if training a segmentation or denoising task)
│   ├── subject_2
│   │   ├── subject_2_input.npy
│   │   ├── subject_2_output.npy (if training a segmentation or denoising task)
│   ├── subject_3
│   │   ├── subject_3_input.npy
│   │   ├── subject_3_output.npy (if training a segmentation or denoising task)
│   ├── dataset_name_metadata.csv (if training a classification task)
```

The ```dataset_name``` and ```subject_IDs``` can be chosen by the user. The path to ```dataset_name``` will be passed as an argument to run.py.

Each subject folder ```<subject_ID>``` needs to have a numpy file named ```<subject_ID>_input.npy```. This represents the network input image. Each numpy file should be formatted as an array of shape ```X Y Z C```, where Z and C are optional dimensions (i.e., they can be squeezed for 2D or single-channel tasks). 

If training a segmentation or denoising task, a numpy file named ```<subject_ID>_output.npy``` should also be included in each subject's folder. This represents the ground truth network output. Similar to the input numpy, each output numpy file should be formatted as an array of shape ```X Y Z C```, where Z and C are optional dimensions (i.e., they can be squeezed for 2D or single-channel tasks). For segmentation, the output file's channel dimension should either be squeezed or have ```C=1```.

If training a classification task, the ```dataset_name``` directory also needs to have a csv of labels called ```dataset_name_metadata.csv```, formatted as follows:

| SubjectID      | Label |
| ----------- | ----------- |
| subject_1      | 0       |
| subject_2   | 1        |
| subject_3   | 1        |

where the SubjectIDs match the naming convention of the data directories, and ```Label``` provides the classification label for that subject.

#### Data splits

If you want to specify which subject is assigned to which split, you will also need to create a file called ```dataset_name_split.csv```, formatted as follows:

| SubjectID      | Split |
| ----------- | ----------- |
| subject_1      | train       |
| subject_2   | val        |
| subject_3   | test        |

The path to ```dataset_name_split.csv``` will be passed as an argument to run.py. 

If you do not specify a dataset split, the dataset will be randomly split into 60% training, 20% val, 20% test.

## Experiments

#### Running experiments

We provide bash files for each of the public datasets in the ```projects``` directory. These bash files provide examples for how to train a model. 

Below, we list the main arguments that need to be modified by each new user: 
  * *wandb_entity*. String that specifies the wandb entity you want to log to; you may also need to run a ```wandb init``` in your cloned directory.
  * *data_dir*. String that specifies the path to the ```dataset_name``` directory where your data is stored.
  * *split_csv_path*. String that specifies the path to the ```dataset_name_split.csv``` file, if using, which specifies splits into train, val, and test.
                                                                    
Below, we list the main arguments that need to be modified to run the experiments reported in the paper: 
  * *encoder_name*. String that specifies the backbone model to use; set to either "ViT" or "Swin".
  * *decoder_name*. String that specifies the decoder, or task head, to use; set to one of "ViTLinear", "SwinLinear", "ViTUNETR", "SwinUNETR", "UperNet2D",  or "UperNet3D" depending on which task you are training.
  * *ViT.patch_size* or *Swin.patch_size*. Int setting the patch size to use for tokenizing the image.
  * *Swin.window_size*. Int setting the window size to use for the local attention window in Swin.
  * *ViT.use_hyena* or *ViT.use_mamba*. Booleans that determine whether to use Hyena or Mamba in place of attention if training a ViT model.
  * *Swin.use_hyena* or *Swin.use_mamba*. Booleans that determine whether to use Hyena or Mamba in place of attention if training a Swin model.
  * *lr*. The learning rate to use.
  * *batch_size*. The batch size to use. 

Additional arguments need to be changed if using a custom dataset (e.g., height, width, no_in_channels). These arguments can be found by looking at the example bash files, although full argument descriptions can be obtained by running ```python run.py --help``` or looking at the files in ```setup/parsers```.

Run an experiment by calling the bash file, for example:
```
bash run_micro.sh
```

#### Results
Results&mdash;including model checkpoints, predictions on the test set, the config, and the log file&mdash;will be stored in the ```logs``` directory. There will also be text files saved with approximate performance metrics, however these metrics are computed as averages over each batch. Recompute metrics using the saved test set predictions to get exact metrics.

## Citations
If you find this study/code useful, please cite our ML4H 2024 Paper, A Study on Context Length and Efficient Transformers for Biomedical Image Analysis. 

## Acknowledgements
We thank the authors of the following repositories, which we built upon in this work.
  * https://github.com/Project-MONAI/MONAI
  * https://github.com/NVlabs/MambaVision
  * https://github.com/state-spaces/mamba
  * https://github.com/HazyResearch/safari
  * https://github.com/HazyResearch/zoology



