cd ../MEDL

nohup python main.py --device 'cuda:0' > ../scripts/log.txt 2>&1 &
