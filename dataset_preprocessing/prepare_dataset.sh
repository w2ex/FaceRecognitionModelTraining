mkdir ~/exp/triplet_training/venv_train_set
virtualenv ~/exp/triplet_training/venv_train_set/experiment
source ~/exp/triplet_training/venv_train_set/experiment/bin/activate
pip install -r requirements_dataset.txt

python preprocess_dataset.py /rex/ssd/tpetit/VGGFace2/test/ /rex/ssd/tpetit/VGGFace2_test_processed_bis/

deactivate
cd ~/exp/triplet_training
rm -r venv_train_set
