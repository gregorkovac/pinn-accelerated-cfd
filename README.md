<h1 align="center">
<span style="font-size:0.6em; font-weight:normal;">
Gregor Kovač
</span>
<br>
Accelerating numerical methods for solving Navier-Stokes equations with physics-informed machine learning
<br>
<span style="font-size:0.6em; font-style:italic; font-weight:normal;">
Master's thesis for the Data Science program at the Faculty of Computer and Information Science, University of Ljubljana
</span>
</h1>

## Current best training params

```sh
python .\train.py --model poisson --data_path ..\..\..\data_glycerol_large --data_samples 10 --samples_per_case 100 --model_path ..\..\models\ --epochs 10000 --weight1 1 --weight2 0 --weight3 0 --weight4 0 --num_workers 0 --use_scheduler True --scheduler_step 200 --scheduler_gamma 0.8 --architecture encdec --pooling avg --activation relu --norm none --upsampling upsample --lr 0.001 --optimizer adam --batch_size 64 --num_layers 5 --hidden_dim 64
```