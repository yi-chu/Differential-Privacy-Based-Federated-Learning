# # communication test

# python3 -u main.py --dataset cifar --model cnn --dp_mechanism NISS --k 0.5 --epochs 100 --local_epochs 10 --num_users 25 --serial --serial_bs 64 --lr 0.1 --iid 
# python3 -u main.py --dataset cifar --model cnn --dp_mechanism NISS --k 0.5 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 

# python3 -u main.py --dataset cifar --model cnn --dp_mechanism DpSecureAggregation --k 0.5 --epochs 100 --local_epochs 10 --num_users 25 --serial --serial_bs 64 --lr 0.1 --iid 
# python3 -u main.py --dataset cifar --model cnn --dp_mechanism DpSecureAggregation --k 0.5 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 

# # accuracy test

# python3 -u main.py --dataset cifar --model cnn --dp_mechanism NISS --k 0.5 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 
# python3 -u main.py --dataset cifar --model cnn --dp_mechanism NISS --k 0.7 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 
# python3 -u main.py --dataset cifar --model cnn --dp_mechanism NISS --k 0.9 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 

# python3 -u main.py --dataset cifar --model cnn --dp_mechanism DpSecureAggregation --k 0.5 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 
# python3 -u main.py --dataset cifar --model cnn --dp_mechanism DpSecureAggregation --k 0.7 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 
# python3 -u main.py --dataset cifar --model cnn --dp_mechanism DpSecureAggregation --k 0.9 --epochs 100 --local_epochs 10 --num_users 50 --serial --serial_bs 64 --lr 0.1 --iid 

# mnist LeNet-5
python3 -u main.py --dataset mnist --model LeNet5 --dp_mechanism NISS --k 0.5 --epochs 30 --local_epochs 10 --num_users 50 
python3 -u main.py --dataset mnist --model LeNet5 --dp_mechanism NISS --k 0.7 --epochs 30 --local_epochs 10 --num_users 50 
python3 -u main.py --dataset mnist --model LeNet5 --dp_mechanism NISS --k 0.9 --epochs 30 --local_epochs 10 --num_users 50 
