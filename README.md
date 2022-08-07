# XAI Thyroid #

## This repository contains the source code for XAI Thyroid -- an XAI object detection problem ##

1. **Install environment**

```
pip install -r requirements.txt
```

2. **Instruct the parameters to be run with each algorithm**

```
python main.py --help
```

3. **Command line example with algorithms**

- **GradCAM**

``` 
python main.py --config_file xAI_config.json --method GradCAM --image_path data/test_images/ --output results/
```

In second stage:

```
python main.py --config_file xAI_config.json --method GradCAM --image_path data/test_images/ --stage second_stage --output results/
```

- **GradCAM++**

```
python main.py --config_file xAI_config.json --method GradCAM++ --image_path data/test_images/ --output results/
```

In second stage:

```
python main.py --config_file xAI_config.json --method GradCAM++ --image_path data/test_images/ --stage second_stage --output results/

```

- **Backpropagation methods**

+ DeepLIFT
+ Saliency
+ grad*input
+ epsilon-LRP
+ intgrad

```
python main.py --config_file xAI_config.json --method elrp --image_path data/test_images/ --output results/
python main.py --config_file xAI_config.json --method saliency --image_path data/test_images/ --output results/
```

- **RISE**

```
python main.py --config_file xAI_config.json --method RISE --image_path data/test_images/ --stage second_stage --output results/
```

- **LIME**

```
python main.py --config_file xAI_config.json --method LIME --image_path data/test_images/ --stage second_stage --output results/ 
```

**Note:** To change input, change the path to new data and path to xml file in xAI_config.json

