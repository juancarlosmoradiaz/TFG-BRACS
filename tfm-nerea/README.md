# BRACS Classification
 Documents of my research in BRACS classification


# Create patches

python create_patches.py --patch_size 512

# Create datasets

python create_datasets.py --patch_size 512

# Train model

python train_RoI_sintest.py --lr 0.01 --epochs 50 --bool_lr_scheduler 1 --results_folder_name resultados_512 --max_patches 5 --batch_size 32 --patch_size 512 --data_RoI data_RoI_512.pkl --data_augmentation 1 --weightsbyclass 1 --dropout 0.2 
