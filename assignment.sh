cd /e/Projects/_neural_network_
python -m pip install opencv-python
python tools/preprocess_brain_mri.py --input data/brain_mri --output data/brain_mri_pgm --size 64 --train-ratio 0.8
python tools/preprocess_brain_mri.py --input data/brain_mri --output data/brain_mri_pgm --size 64 --image data/brain_mri/yes/Y1.jpg --image-out data/brain_mri_pgm/single_image.pgm
cmake --preset default -DNN_BUILD_BRAIN_MRI=OFF -DNN_BUILD_BRAIN_MRI_PGM=ON -DNN_FORCE_CPU=OFF
cmake --build --preset default --config Release --target brain_mri_pgm
./build/default/Release/brain_mri_pgm.exe --train-manifest data/brain_mri_pgm/train_manifest.txt --test-manifest data/brain_mri_pgm/test_manifest.txt --size 64 --epochs 30 --lr 0.005 --save checkpoints/brain_mri_e30_lr5e3.ckpt
./build/default/Release/brain_mri_pgm.exe --train-manifest data/brain_mri_pgm/train_manifest.txt --test-manifest data/brain_mri_pgm/test_manifest.txt --size 64 --load checkpoints/brain_mri_e30_lr5e3.ckpt --epochs 10 --lr 0.001 --save checkpoints/brain_mri_resume.ckpt
./build/default/Release/brain_mri_pgm.exe --train-manifest data/brain_mri_pgm/train_manifest.txt --test-manifest data/brain_mri_pgm/test_manifest.txt --size 64 --load checkpoints/brain_mri_resume.ckpt --epochs 0 --lr 0.001

# To be shown
./build/default/Release/brain_mri_pgm.exe --load checkpoints/brain_mri_resume.ckpt --infer-only --size 64 --image data/brain_mri_pgm/single_image.pgm