# COSPRO-mix

This is the official repository for "Exploring the Impacts of Phonetics, Prosody, and Languages in Speech Separation Networks". We provided the script of how we mixed COSPRO-2mix and COSPRO-3mix in addition to some of the test sets. 

## Instructions
First, we have to prepare the environment for data-mixing.

    git clone https://github.com/Sinica-SLAM/COSPRO-mix
    cd COSPRO-mix
    pip install -r requirements.txt

Then, we can mix the data we want by running `create_mix_file.py`. For instance, if you want to create COSPRO-2mix, you can run the command below.

    python create_mix_file.py -o <output dir> -c <COSPRO root>

There are also some options for other data. All of the arguments are listed below.

1. `-o`: the output directory to put the mixed data
2. `-c`: the root directory of COSPRO
3. `-w`: the root directory of WSJ0 if needed
4. `-n`: the number of speakers in each utterance.
5. `-d`: the dataset you want to mix

## Benchmarks

Below, we provided the results of some representative speech separation models trained on COSPRO-2mix and COSPRO-3mix for 100 epochs.

- COSRPO-2mix (SI-SNR / SI-SNRi)
    | | dev | test |
    | - | - | - |
    | Conv-TasNet | 17.10 / 17.06 | 15.95 / 15.91 |
    | DPRNN | 18.06 / 18.02 | 16.96 / 16.92 |
    | DPT-Net | 18.15 / 18.11 | 17.27 / 17.23 |

- COSPRO-3mix (SI-SNR / SI-SNRi)
    | | dev | test |
    | - | - | - |
    | Conv-TasNet | 9.36 / 12.52 | 7.75 / 10.92 |
    | DPRNN | 10.73 / 13.89 | 9.33 / 12.50 |
    | DPT-Net | 11.08 / 14.24 | 10.05 / 13.22 |
