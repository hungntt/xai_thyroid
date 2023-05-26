![img_2.png](img_2.png)
# Towards XAI in Thyroid Tumor Diagnosis #
Our paper is accepted and presented as a long presentation at Health Intelligence Workshop (W3PHIAI-23) at AAAI-23.
## Source code for XAI Thyroid - an XAI object detection problem ##

1. **Install environment**

```
pip install -r requirements.txt
```

- Download the model at: https://drive.google.com/file/d/1IOyom78mexC6BPq4gPkfFLg-3vPzBnnO/view?usp=sharing

Move the model to ``model/src/`` folder.

2. **Instruct the parameters to be run with each algorithm**

```
python main.py --help
```

3. **Command line example with algorithms**
   Arguments options:

- `--config-path`: path to the configuration file
- `--method`: XAI method to run (options: eLRP, GradCAM, GradCAM++, RISE, LIME, DRISE, KDE, DensityMap, AdaSISE)
- `--image-path`: path to the image to be processed
- `--stage`: stage of the algorithm to be run (options: first_stage, second_stage, default: first_stage)
- `--threshold`: threshold of output values to visualize
- `--output-path`: path to the output directory

For example, to run the XAI algorithms on images in test_images folder:

- **GradCAM**

In first stage:

``` 
python main.py --config-path xAI_config.json --method GradCAM --image-path data/test_images/ --output-path results/
```

In second stage:

```
python main.py --config-path xAI_config.json --method GradCAM --image-path data/test_images/ --stage second_stage --output-path results/
```

- **GradCAM++**

In first stage:

```
python main.py --config-path xAI_config.json --method GradCAM++ --image-path data/test_images/ --output-path results/
```

In second stage:

```
python main.py --config-path xAI_config.json --method GradCAM++ --image-path data/test_images/ --stage second_stage --output-path results/

```

**Note:** To change input, change the path to new data and path to xml file in xAI_config.json

## Applicability
![img_1.png](img_1.png)
• Region Proposal Generation (Which proposals are generated by the model during the model’s first stage?): Kernel Density Estimation (KDE), Density map (DM).

• Classification (Which features of an image make the model classify an image containing a nodule(s) at the model’s second stage?): LRP, Grad-CAM, Grad-CAM++, LIME, RISE, Ada-SISE, D-RISE.

• Localization (Which features of an image does the model consider to detect a specific box containing a nodule at the model’s second stage?): D-RISE.
## Results 
![img.png](img.png)

## Citation
If you find this repository helpful for your research. Please cite our paper as a small support for us too :)
```
@article{nguyen2023towards,
  title={Towards Trust of Explainable AI in Thyroid Nodule Diagnosis},
  author={Nguyen, Truong Thanh Hung and Truong, Van Binh and Nguyen, Vo Thanh Khang and Cao, Quoc Hung and Nguyen, Quoc Khanh},
  journal={arXiv preprint arXiv:2303.04731},
  year={2023}
}
```
