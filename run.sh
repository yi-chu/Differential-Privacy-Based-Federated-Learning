# python3 -u main.py --dataset mnist --model cnn --dp_mechanism NISS --k 0.7 --epochs 100 --num_users 100 --dropout 10
# python3 -u main.py --dataset mnist --model cnn --dp_mechanism NISS --k 0.7 --epochs 100 --num_users 100 --dropout 20
# python3 -u main.py --dataset mnist --model cnn --dp_mechanism NISS --k 0.7 --epochs 100 --num_users 100 --dropout 30
# python3 -u main.py --dataset mnist --model cnn --dp_mechanism DpSecureAggregation --k 0.7 --epochs 100 --num_users 100 --dropout 10
# python3 -u main.py --dataset mnist --model cnn --dp_mechanism DpSecureAggregation --k 0.7 --epochs 100 --num_users 100 --dropout 20
# python3 -u main.py --dataset mnist --model cnn --dp_mechanism DpSecureAggregation --k 0.7 --epochs 100 --num_users 100 --dropout 30

python3 -u main.py --dataset mnist --model cnn --dp_mechanism MA --epochs 100 --num_users 25
python3 -u main.py --dataset mnist --model cnn --dp_mechanism MA --epochs 100 --num_users 50

python3 -u main.py --dataset mnist --model cnn --dp_mechanism MA --k 0.7 --epochs 100 --num_users 100 --dropout 10
python3 -u main.py --dataset mnist --model cnn --dp_mechanism MA --k 0.7 --epochs 100 --num_users 100 --dropout 20
python3 -u main.py --dataset mnist --model cnn --dp_mechanism MA --k 0.7 --epochs 100 --num_users 100 --dropout 30