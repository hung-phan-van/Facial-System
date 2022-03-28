
fromFolder=/home/ailab/users/hung/allstaff
toFolder=/home/ailab/users/hung/dataset_2
mlFolder=/home/ailab/users/hung/face_system/face_recognition

echo $toFolder
echo $fromFolder
rm -rf $toFolder/allstaff
scp -r $fromFolder $toFolder
cd $toFolder/allstaff
rm -rf 00Unknown

cd $mlFolder
python3 -u embedding.py --data-dir $toFolder/allstaff --model models/model-r100 --model-epoch 23 --gpu 1
python3 -u classifier.py --data-dir $toFolder/allstaff --model models/model-r100 --model-epoch 23 --model-type ann --name all_ann_ds2 --gpu 1
echo "DONE ALL AND STARTING AGAIN"
