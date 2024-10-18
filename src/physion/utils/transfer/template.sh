# folder to transfer
to_transfer='~/UNPROCESSED/YANN/2024_10_10'
# storage informations
nas='user@10.0.0.1'
user_folder='Yann'
dataset = 'SST-Ketamine-WT'
# 
# transfert to NAS of ** PROCESSED ** imaging/camera data
rsync -avhP $to_transfer $nas:/volume1/$user_folder/$dataset/Raw/ --dry-run # for debugging
# transfert to NAS of ** RAW ** imaging/camera data
rsync -avhP $to_transfer $nas:/volume1/$user_folder/$dataset/Processed/ --dry_run # for debuggin
