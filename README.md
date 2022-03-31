# COSPRO-mix

This is the official repository for "Disentangling the Impacts of Language and Channel Variability on Speech Separation Networks". We provided the script of how we mixed COSPRO-2mix and TAT-mix in addition to some of the test sets. 

## Instructions
First, we have to prepare the environment for data-mixing.

    git clone https://github.com/Sinica-SLAM/COSPRO-mix
    pip install -r requirements.txt

Then, we can mix the data we want by running `create_mix_file.py`. For instance, if you want to create COSPRO-2mix, you can run the command below.

    python create_mix_file.py -o <output dir> -c <COSPRO root>

There are also some options for other data:
1. `-t`: Path to the folder containing TAT wavs, needed when creating TAT-mix.
2. `-w`: Path to the folder containing WSJ0 dirs, needed when creating WSJ0$\times$COSPRO.
3. `-n`: Mix 2 or 3 speaker. Default is set to 2.
4. `-f`: Whether make different overlap ratios ,True for only fully overlapped (Default: True)
5. `-d`: The data you want to mix. Other datasets can be referred in data folder.