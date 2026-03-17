<h1 align="center">
<br>
Accelerating numerical methods for solving Navier-Stokes equations with physics-informed machine learning
<br>
</h1>

## Current best training params

### Poisson PINN
```sh
python .\train.py --model poisson --data_path ..\..\..\data_glycerol_large --data_samples 100 --samples_per_case 300 --model_path ..\..\models\ --epochs 10000 --weight1 1 --weight2 0 --weight3 0 --weight4 0 --num_workers 0 --use_scheduler True --scheduler_step 500 --scheduler_gamma 0.8 --architecture encdec --pooling stride --activation relu --norm none --upsampling upsample --lr 0.001 --optimizer adam --batch_size 128 --num_layers 5 --hidden_dim 8 --skips unet --plot False
```

### Full PINN
```sh
python .\train.py --model poisson --data_path ..\..\..\data_glycerol_large --data_samples 100 --samples_per_case 300 --model_path ..\..\models\ --epochs 10000 --weight1 1 --weight2 0 --weight3 0 --weight4 0 --num_workers 0 --use_scheduler True --scheduler_step 500 --scheduler_gamma 0.8 --architecture encdec --pooling stride --activation relu --norm none --upsampling upsample --lr 0.001 --optimizer adam --batch_size 128 --num_layers 5 --hidden_dim 8 --skips unet --plot False
```