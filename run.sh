nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 500 --depth 1 &
nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 500 --depth 3 &
nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 500 --depth 5 &
wait
nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 500 --depth 7 &
nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 500 --depth 9 &
wait 
nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 10000 --depth 1 &
nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 2000 --depth 3 &
wait 
nohup python main_train.py --dataset mnist --arch fnn --drop 20 30 --nepochs 35 --width 2000 --depth 5 &
