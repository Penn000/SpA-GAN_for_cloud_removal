# SpA-GAN_for_cloud_removal

## 1. INTRODUCTION

**Generator**

See `./models/gen/SPANet.py` for more details.

![SPANet](D:/Heng/Documents/workspace/SpA-GAN_for_cloud_removal/readme_images/SPANet.jpg)

**Discriminator**

Discriminator is a fully  CNN that **C** is convolution layer, **B** is batch normalization and **R** is Leaky ReLU. See `./models/dis/dis.py` for more details.

![dis](D:/Heng/Documents/workspace/SpA-GAN_for_cloud_removal/readme_images/dis.jpg)

**Loss**

The total loss of *SpA GAN* is formulated as fellow:

![loss_spa-gan](./readme_images/loss_spagan.png)

and

![loss_cgan](./readme_images/loss_cgan.png)

![loss_l1](./readme_images/loss_l1.png)

![loss_att](./readme_images/loss_att.png)

## 2. DATASET

### 2.1. RICE_DATASET

Click [here](https://github.com/BUPTLdy/RICE_DATASET) to download the open source RICE dataset. Build the file structure as the folder `data` shown. Here `cloudy_image` is the folder where the cloudy image is stored and the folder `ground_truth` stores the corresponding cloudless images.

```
./
+-- data
	+--	RICE_DATASET
		+--	RICE1
		|	+-- cloudy_image
		|	|	+-- 0.png
		|	|	+-- ...
		|	+-- ground_truth
		|		+-- 0.png
		|		+-- ...
		+--	RICE2
			+-- cloudy_image
			|	+-- 0.png
			|	+-- ...
			+-- ground_truth
				+-- 0.png
				+-- ...
```

### 2.2. Perlin Dataset

Construct the dataset by adding Perlin noise as cloud into the image.

## 3. TRAIN

Modify the `config.yml` to set your parameters and run:

```bash
python train.py
```

## 4. TEST

```bash
python predict.py --config <path_to_config.yml_in_the_out_dir> --test_dir <path_to_a_directory_stored_test_data> --out_dir <path_to_an_output_directory> --pretrained <path_to_a_pretrained_model> --cuda
```

