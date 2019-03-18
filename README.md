# Strain Identification

# Installations:
a.) Numpy
To install numpy -  https://scipy.org/install.html

b.) Bio Python
To install Bio Python - https://biopython.org/wiki/Download

# How to Use:

Training -

IBNB_Train.py will be used for training the model. This will train the model on the given reference classes and store that model.

To get details about different parameters on command line:

'python3 IBNB_Train.py --help'  
or  
'python3 IBNB_Train.py -h'


To execute from command line:

'python3 IBNB_Train.py --refpath=REFPATH --klen=KLEN --hklen=HKLEN --cores=CORES --output_model=OUT'

REFPATH : path of the folder containing reference genomes on which you want to train your model   
KLEN    : size of the word , default = 16    
HKLEN   : size of the subword , default = 8    
CORES   : number of cores you want to use for training on multiple references      
OUT     : path where you want to store your model     



Testing - 

IBNB_Test.py will be used for testing the strains. It will use the pretrained model for calculating the final score.

To get details about different parameters on command line:

'python3 IBNB_Train.py --help'  
or  
'python3 IBNB_Train.py -h'

To execute from command line:

'python3 IBNB_Test.py --testpath=TESTPATH --reference_interval=INTERVAL --cores=CORES --input_model=IN'


TESTPATH  : path of the folder containing test samples
INTERVAL  : Number of reads after which you want the predicted result.
CORES     : number of cores you want to use for testing on multiple test samples
IN        : path of pretrained model





# More details will be added soon.
