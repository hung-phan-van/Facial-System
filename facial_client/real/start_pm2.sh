eval "$(conda shell.bash hook)"
conda activate face_client
pm2 delete  help_restart client_sent_image
pm2 start help_restart.sh client_sent_image.py --time --no-autorestart
