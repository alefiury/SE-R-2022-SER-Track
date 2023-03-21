mkdir -p data
cd data

# CORAA SER train set
gdown --id 1N56YOgJ_plF4K8Eyh9hqiP0_O5L8uwya
unzip data_train.zip
mv train CORAA_SER
rm data_train.zip

# EMOVO
gdown --id 1SUtaKeA-LYnKaD3qv87Y5wYgihJiNJAo
unzip emovo.zip
mv emovo EMOVO
rm emovo.zip

# BAVED
git clone https://github.com/40uf411/Basic-Arabic-Vocal-Emotions-Dataset
mv Basic-Arabic-Vocal-Emotions-Dataset BAVED

# RAVDESS
wget -c https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip
mkdir -p RAVDESS
mv Audio_Speech_Actors_01-24.zip RAVDESS
cd RAVDESS
unzip Audio_Speech_Actors_01-24.zip
rm Audio_Speech_Actors_01-24.zip

cd ..
mkdir -p train
mv CORAA_SER train
mv EMOVO train
mv BAVED train
mv RAVDESS train

# CORAA SER test set
gdown --id 1UQOi59rFbk5bVvaF7lQWFwRaae8eaOKV
unzip test_ser.zip
mv test_ser 'test'
rm test_ser.zip

# CORAA SER test set metadata
cd ..
mkdir -p metadata
cd metadata
gdown --id 1BQCkD2gKAofIsqmcVp7bB2sauqwVietz