# COSPRO-mix

This is the official repository for "Exploring the Impacts of Phonetics, Prosody, and Languages in Speech Separation Networks". We provided the script of how we mixed COSPRO-2mix and COSPRO-3mix in addition to some of the test sets. 

## Instructions
First, we have to prepare the environment for data-mixing.

    git clone https://github.com/Sinica-SLAM/COSPRO-mix
    pip install -r requirements.txt

Then, we can mix the data we want by running `create_mix_file.py`. For instance, if you want to create COSPRO-2mix, you can run the command below.

    python create_mix_file.py -o <output dir> -c <COSPRO root>

There are also some options for other data. All you need to do is to modify `-d` to "data_3mix". Other datasets can be referred in data folder.