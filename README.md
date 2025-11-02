# CXR-classification

## Classification categories:  
```
1. Normal
2. Bacteria (Pneumonia)
3. Virus (Pneumonia)
4. COVID-19
```
## Environments  
```bash
conda create -n CXR python=3.12 -y
conda activate CXR
```
```bash
pip install -q uv
uv pip install -r requirements.txt
```

## Check Std & Mean of dataset
> Put std and mean value into dataloader
```
python cal_mean_std.py
```  

## Training
> Begin training the model, then evaluate it once training is finished
```
python main.py
```
> Or, you can specify a configuration for training for example:
```
python main.py --backbone efficientnet_v2_m --img-size 480 --epochs 300 --bs 48 --lr 5e-4 --weight-decay 5e-4 --dropout 0.25 --loss combined --focal-gamma 1.5 --cb-beta 0.999 --label-smoothing 0.02 --sched cosine --warmup-epochs 10 --min-lr 1e-5 --es-patience 30 --use-tta --tta-times 4 --save-probs --out-csv Submission.csv
```

## Testing
> Testing will automatically run after training is completed. If you want to perform testing only, you can run:
```
python main.py --test-only --backbone efficientnet_v2_s --ckpt-path best_efficientnet_v2_s.pt --img-size 384 --bs 32 --test-dir ./datasets/test_images --use-tta --tta-times 5 --save-probs --out-csv Submission.csv
```

