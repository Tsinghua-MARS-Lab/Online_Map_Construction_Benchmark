# Online HD Map Construction Benchmark for Autonomous Driving

This is a benchmark for the online map construction task. This task aims to construct the local HD map from onboard sensor observations (cameras images). 

![surrounding_images](https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/resources/surrounding_images.png)

Typically, local maps can be represented by rasterized output or vectorized output. We set benchmark for two tasks termed rasterized map construction and vectorized map construction.

#### Rasterized map construction

The goal of rasterized map construction is to output a segmentation image of bird's-eye-view for each sample. Each pixel on the rasterized image corresponds to a grid on ground surface. Here is an example.

![semantic_map](https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/resources/semantic_map.jpg)

We use Intersection over Union (IoU) as metric to evaluate the quality of construction.

#### Vectorized map construction

The goal of vectorized map construction is to output a sparse set of polylines. Polylines serve as primitives to model the geometry of maps elements. Here is an example.

<img src="https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/resources/vectorized_map.jpg" width = "400" height = "200" alt="vectorized_map"/>

We use Chamfer Distance based Average Precision (AP) as metric to evaluate the quality of construction as introduced in [HDMapNet](https://arxiv.org/abs/2107.06307) and [VectorMapNet](https://arxiv.org/abs/2206.08920). For details of evaluation metrics, please see [Evaluation.md](https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/resources/Evaluation.md).

Currently we build our benchmark on the [nuScenes](www.nuscenes.org) dataset. We include three types of map elements: pedestrian crossing, lane divider and road boundary. We provide baseline models, data processing pipelines and evaluation kit for both rasterized and vectorized tasks. To the best of our knowledge, this is the first online local map construction benchmark. We hope this benchmark could aid future research on online local map construction.



## Preparation

#### 1. Environment Preparation

**Step 1.** Create conda environment and activate it.

```
conda create --name hdmap python==3.8
conda activate hdmap
```

**Step 2.** Install PyTorch.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```

**Step 3.** Install MMCV series.

```
# Install mmcv-series
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**Step 4.** Install mmdetection3d.

Currently we are usingmmdetection3d of  version 0.17.3 . To install mmdet3d, please first download the releases of 0.17.3 from https://github.com/open-mmlab/mmdetection3d/releases. Then run

```
cd mmdetection3d-0.17.3
pip install -v -e .
```

For more details about installation, please refer to open-mmlab [getting_started.md](https://github.com/open-mmlab/mmdetection3d/blob/master/docs/en/getting_started.md).

**Step 5.** Install other requirements.

```
pip install -r requirements.txt
```

#### 2. Data Preparation

**Step 1.** Download [nuScenes](www.nuscenes.org) dataset and assume the path is `/path/to/nuScenes/`.

**Step 2.** Make a softlink under `dataset` folder.

```
mkdir datasets
ln -s /path/to/nuScenes/ ./datasets/nuScenes
```

**Step 3.** Generate annotation files

```
python tools/data_converter/nuscenes_converter.py --data-root ./datasets/nuScenes
```



## Baseline Models

We provide a baseline model for each of the two tasks: [HDMapNet](https://arxiv.org/abs/2107.06307) for rasterized construction and [VectorMapNet](https://arxiv.org/abs/2206.08920) for vectorized construction.

#### Expected Results

Rasterized model

|      | Ped Crossing | Divider | Boundary | Mean  |                         Config File                          | Ckpt |
| :--: | :----------: | :-----: | :------: | :---: | :----------------------------------------------------------: | :--: |
| IoU  |     20.2     |  38.13  |   35.6   | 31.31 | [config](https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/plugin/configs/raster/semantic_nusc.py) |      |

Vectorized model

|      | Ped Crossing | Divider | Boundary | Mean  |                         Config File                          | Ckpt |
| :--: | :----------: | :-----: | :------: | :---: | :----------------------------------------------------------: | :--: |
|  AP  |    38.87     |  49.23  |  38.15   | 42.09 | [config](https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/plugin/configs/vector/vectormapnet_nusc.py) |      |

#### Training

Single GPU training

```
python tools/train.py ${CONFIG_PATH}
```

For example

```
python tools/train.py plugin/configs/raster/semantic_nusc.py
```

Multi GPUs training

```
bash tools/dist_train.sh ${CONFIG_PATH} ${NUM_GPUS}
```

For example

```
bash tools/dist_train.sh plugin/configs/vector/vectormapnet_nusc.py 8
```



## Evaluation

For details of evaluation metrics, please see [Evaluation.md](https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/resources/Evaluation.md).

#### To evaluate a checkpoint

Single GPU evaluation

```
python tools/test.py ${CONFIG_PATH} --checkpoint ${CHECKPOINT} --eval
```

For example

```
python tools/test.py plugin/configs/vector/vectormapnet_nusc.py --checkpoint work_dirs/vectormapnet_nusc/latest.pth --eval
```

Multi GPUs evaluation

```
bash tools/dist_test.sh ${CONFIG_PATH} ${CHECKPOINT} ${NUM_GPUS} --eval 
```

#### To evaluate a submission file

```
python tools/test.py ${CONFIG_PATH} --result-path ${RESULT_PATH} --eval
```

For example

```
python tools/test.py plugin/configs/vector/vectormapnet_nusc.py --result-path work_dirs/vectormapnet_nusc/submission_vector.pkl --eval
```

Evaluating a checkpoint will automatically generate a submission file. Or you can generate one by running

```
python tools/test.py ${CONFIG_PATH} --checkpoint ${CHECKPOINT} --format-only
```



## Visualization

Visualization tools can be used to visualize ground-truth labels and model's prediction results. For rasterized model, only a rasterized BEV segmentation image will be generated. For vectorized model, map labels are projected on surrounding camera images and six images will be saved. 

To visualize ground-truth labels

```
python tools/visualization/visualize.py ${CONFIG_PATH} ${DATA_IDX}
```

To visualize both ground-truth and prediction labels

```
python tools/visualization/visualize.py ${CONFIG_PATH} ${DATA_IDX} --result ${RESULT_PATH}
```

For example

```
python tools/visualization/visualize.py \
	plugin/configs/vector/vectormapnet_nusc.py \
	5 \
	--result work_dirs/vectormapnet_nusc/submission_vector.pkl
```



## Acknowledgement

This project is built on [MMCV](https://github.com/open-mmlab/mmcv), [MMDetection](https://github.com/open-mmlab/mmdetection), [MMDetection3D](https://github.com/open-mmlab/mmdetection3d) frameworks. We would like thank [Motional](https://motional.com/) for providing [nuScenes](https://nuscenes.org/) dataset and allowing us to build this benchmark on it.



## License

This project is released under [GNU General Public License v3.0](https://github.com/Tsinghua-MARS-Lab/Online_Map_Construction_Benchmark/blob/master/LICENSE).



## Citation

If you find this project useful in your research, please cite:

```
@article{liu2022vectormapnet,
    title={VectorMapNet: End-to-end Vectorized HD Map Learning},
    author={Liu, Yicheng and Wang, Yue and Wang, Yilun and Zhao, Hang},
    journal={arXiv preprint arXiv:2206.08920},
    year={2022}
    }
@article{li2021hdmapnet,
    title={HDMapNet: An Online HD Map Construction and Evaluation Framework},
    author={Qi Li and Yue Wang and Yilun Wang and Hang Zhao},
    journal={arXiv preprint arXiv:2107.06307},
    year={2021}
}
```



## Contributors

The main contributors of this project are [Tianyuan Yuan](https://github.com/yuantianyuan01) and [Yicheng Liu](https://github.com/Mrmoore98). If you have any questions or requests, please raise an issue or send an e-mail to yuantianyuan01@gmail.com.

