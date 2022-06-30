CLASSES=50
SAMPLES=30
DIM=128
EPOCHS=60
GPU=0
mkdir ~/exp/triplet_training/venv_${CLASSES}_${SAMPLES}_${DIM}_${GPU}
virtualenv ~/exp/triplet_training/venv_${CLASSES}_${SAMPLES}_${DIM}_${GPU}/experiment
source ~/exp/triplet_training/venv_${CLASSES}_${SAMPLES}_${DIM}_${GPU}/experiment/bin/activate
pip install -r requirements.txt
python3 training_script_triplet.py /rex/ssd/tpetit/VGGFace2_processed_bis/ /home/tpetit/exp/results_${CLASSES}classes_${SAMPLES}samples_inception_rn /rex/ssd/tpetit/lfw_processed/ -m /rex/ssd/tpetit/models/model_${CLASSES}classes_${SAMPLES}samples_rn18 -a rn18 -e $EPOCHS -c $CLASSES -s $SAMPLES -d $DIM -g $GPU
deactivate
cd ~/exp/triplet_training
rm -r venv_${CLASSES}_${SAMPLES}_${DIM}_${GPU}
