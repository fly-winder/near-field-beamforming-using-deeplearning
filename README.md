# Deep Learning-Based Beamforming for Near-Field Massive MIMO

This repository contains the implementation of our proposed deep learning-based beamforming approach for near-field massive MIMO systems. The model leverages neural networks to optimize beamforming vectors, improving achievable rates while maintaining computational efficiency.

## Repository Structure

- `model.py`: Defines the neural network architecture for beamforming optimization.
- `mymodel.pth`: Pretrained model weights for inference.
- `train_pytorch.py`: Script for training the beamforming model.
- `test_pytorch.py`: Script for testing the trained model and evaluating performance.
- `utils_pytorch.py`: Utility functions for data processing, evaluation, and model support.

## Requirements

Ensure you have the following dependencies installed:

```bash
pip install torch numpy matplotlib
```

## Usage

### Training the Model
To train the model from scratch, run:

```bash
python train_pytorch.py
```

### Testing the Model
To evaluate the trained model, execute:

```bash
python test_pytorch.py
```

### Inference with Pretrained Model
To use the pretrained model for inference, modify `test_pytorch.py` to load `mymodel.pth` and execute the script.

## Citation
If you find this repository useful for your research, please cite our paper:

```
@ARTICLE{10682562,
  author={Nie, Jiali and Cui, Yuanhao and Yang, Zhaohui and Yuan, Weijie and Jing, Xiaojun},
  journal={IEEE Transactions on Mobile Computing}, 
  title={Near-Field Beam Training for Extremely Large-Scale MIMO Based on Deep Learning}, 
  year={2025}
  doi={10.1109/TMC.2024.3462960}
}
```


