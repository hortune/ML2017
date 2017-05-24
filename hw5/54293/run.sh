python3 preprocess.py > /dev/null
python3 0.py train_data.csv  $1 > /dev/null
python3 1.py train_data.csv  $1 > /dev/null
python3 2.py $1 > /dev/null
python3 3.py  > /dev/null
python3 vote.py $2
