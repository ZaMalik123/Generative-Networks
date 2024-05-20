# Adversarial Computation of Optimal Transport Maps
Code built off of Jennifer She's repository here: https://github.com/jshe/wasserstein-2

## Dependencies

```
torch (0.4.1)
numpy (1.14.3)
h5py (2.7.1)
torchvision (0.2.1)
scikit-learn (sklearn) (0.19.1)
matplotlib (2.2.2)
python (3.6)
tensorboardX (optional, remove dependency if not used)
```

## Experiments
All experiments are intended to be run through the command line. I recommend using Google Colab to run this code.

### 2D/OT (exp_2d)

```
# W2-GAN
# 4 gaussians
python main.py --solver=w2 --gen=1 --data=4gaussians
# swissroll
python main.py --solver=w2 --gen=1 --data=swissroll
# checkerboard
python main.py --solver=w2 --gen=1 --data=checkerboard
# rings
python main.py --solver=w2 --gen=1 --data='ring'
# W2-OT
# 4 gaussians
python main.py --solver=w2 --gen=0 --data=4gaussians --train_iters=20000
# swissroll
python main.py --solver=w2 --gen=0 --data=swissroll --train_iters=20000
# checkerboard
python main.py --solver=w2 --gen=0 --data=checkerboard --train_iters=20000
```

```

### Multivariate Gaussian ⟶ MNIST (exp_mvg)

```
python main.py --solver=w2
```

### Domain Adaptation: MNIST ⟷ USPS (exp_da)

```
# usps -> mnist
python main.py --solver=w2 --direction=usps-mnist
# mnist -> usps
python main.py --solver=w2 --direction=mnist-usps
```
## Acknowledgments

* https://github.com/mikigom/large-scale-OT-mapping-TF.git
* https://github.com/igul222/improved_wgan_training.git

