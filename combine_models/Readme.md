#First run this Command for install all dependencies

pip install -r requirements.txt

# run the command to predict the word 
# model support only audio file i.e mp3,wav
# word should belongs to any label which is mention in the list [bed, cat, down, left, no, right, seven, stop, yes, up]
# if you wanna try wih your own file than move your file into data folder and change the 'down' to your file name

python3 main.py --data_path data/down.wav

 

