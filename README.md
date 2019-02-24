# Globally and Locally Consistent Image Completion

Kangzeroo's Keras 2.2.4 Implementation of the Paper ["Globally and Locally Consistent Image Completion"](http://hi.cs.waseda.ac.jp/%7Eiizuka/projects/completion/data/completion_sig2017.pdf). (Satoshi Iizuka et al, Waseda University, Japan)

![Results of Original Paper](readme/preview.png)

## Setup
0. `conda env create -f environment.yml`
1. Spin up GCloud VM and SSH into it
2. Use `lsun_script.sh` to setup VM and download dataset
3. Open `$ ipython notebook` to run through steps making the brain
4. Train network (est. 14 days on VM x1 GPU V80)

## Keras Implementation
<br/><br/>
Generator
![Generator](readme/generator.png)
<br/><br/>
Discriminator
![Discriminator](readme/discriminator.png)
<br/><br/>
Together
![Brain](readme/brain.png)
<br/><br/>

## Architecture
![Generative Adversarial Net](readme/overview.png)
