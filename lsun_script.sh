
# ------------ CREATE AND SSH INTO YOUR CLOUD VM ------------
# updates
sudo apt-get update
# download brew for debian
sudo apt-get brew
# download git
sudo brew install git
# download python2.7 and opencv
sudo git clone https://github.com/fyu/lsun


# ------------ INSIDE YOUR CLOUD VM ------------
# download nohup to allow running scripts in background
# https://janakiev.com/til/python-background/
# sudo apt-get install nohup

# run your python or bash scripts with nohup, & background, and log output, simply by:
# sudo chmod +x test.py
# sudo mkdir logs
# sudo chmod 777 logs
# sudo nohup python test.py > path/to/output.log &

sudo nohup gsutil -m cp -n -r data/bedroom/bedroom_train_lmdb/images gs://lsun-roomsets/images/bedroom_train > logs/bedroom-gbucket.log &

# view output logs
# sudo tail -200 logs/output.log

# alternatively you can run your scripts sequentially if no errors using &&:
# sudo nohup sh -c 'first_command && second_command -f flag' &

# unzip to .mdb files
sudo unzip kitchen_train_lmdb.zip

# remove the old .zip files
sudo rm data/kitchen/kitchen_train_lmdb.zip

# convert from .mdb to images
sudo python2.7 data.py export data/kitchen/kitchen_train_lmdb/ --out_dir data/kitchen/kitchen_train_lmdb/images/

# remove the old .mdb files
sudo rm data/kitchen/kitchen_train_lmdb/data.mdb
sudo rm data/kitchen/kitchen_train_lmdb/lock.mdb

# transfer to gcloud bucket
sudo gsutil -m cp -n -r data/kitchen/kitchen_train_lmdb/images gs://lsun-roomsets/images/kitchen_train

# delete the old images
sudo rm -rf data/kitchen/kitchen_train_lmdb/images

# view running jobs
jobs
