import os
import sys
import time
import pickle 
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
from optparse import OptionParser


class IBNB_Model_test:

    def __init__(self):
        self.no_of_pools=0        
        self.testSamples = []        
        #self.testSamples_reads = []               
        #self.total_testSamples = 0 
        self.testSamples_type = "fastq"               
        self.fastq_testPath = []
        self.partial_test_interval=0          
        self.input_model_path = []                   
            
    def readFastqFiles(self):
        for idx,sample in enumerate(self.testSamples):
            fname = os.path.join(self.fastq_testPath,sample)
            print("Reading Test sample :", fname)            
            self.testSamples_reads.append([str(record.seq) for record in SeqIO.parse(fname,self.testSamples_type)] )    

    def readTestStrains(self):
        print("\n======== Reading the Test Samples ========")
        self.testSamples = os.listdir(self.fastq_testPath)
        self.testSamples.sort()
        self.total_testSamples = len(self.testSamples)         
        print("Total test samples in Fastq Test folder: ",self.total_testSamples )
        self.readFastqFiles()      
        print("==== Reading of Test Samples Completed ====\n")      

    def test(self): 
        self.testSamples_reads = []
        self.readTestStrains()        
        pool = Pool(self.no_of_pools)  
        test_list=list(range(self.total_testSamples))
        testklen_cnt = (pool.starmap(test_pool,
                        [(test_no,
                            self.testSamples_reads[test_no],
                            self.testSamples) for test_no in test_list]))
        pool.close()
        pool.join()


def test_pool(test_no, sample,testSamples):    
    clk_temp = time.time()
    print("\n============Testing of Test Samples started============")
    print("Test sample ID: ", test_no, " ; Test sample name: ",testSamples[test_no])        
    AB_score = np.zeros(total_refStrains)
    hits=np.zeros(total_refStrains)
    all_count=0

    read_count = 0        
    for reads in sample:  
        
        read_count += 1
        all_count += (len(reads)-klen)
        for i in range(len(reads) - klen):
            AB = reads[i:i + klen]            
            if AB in  cond_prob:                   
                AB_score = AB_score + cond_prob[AB] 
                hits = np.add(hits,1,out=hits,where=cond_prob[AB]!=0)                
                 
        if read_count % partial_test_interval == 0 and read_count != 0:            
            print("\n-------------------------START------------------------------------")
            print("Table partial filled for sample ID : ", test_no, " ; Read Count : ", read_count)  
            print("Time taken to partially test Sample ID: ", test_no, " is : ", time.time() - clk_temp)
            print("Intermediate Sorted Score: ")
            print("Rank      Score               Strain Name")
            for idx,(j,k) in enumerate(sorted([(AB_score[i], i) for i in range(total_refStrains)],reverse=True)):
                if idx>9:break                
                print("{:4d}\t{:^13.3f}\t{}".format(idx+1,j,refStrains[k]))
            
            idx_res =np.argmax(AB_score)
            print("\nSample's Original Class is       :- ", testSamples[test_no])                 
            print("Sample's Predicted Class is      :- ",idx_res ,refStrains[idx_res])               
            print("----------------------------END------------------------------------\n")
        
        
    print("\n*************************FINAL RESULTS*************************")
    print("Total time taken to test Sample ID: ", test_no, " is : ", time.time() - clk_temp)
    print("Total processed Read Count : ", read_count)
    print("Test sample name is  : ",testSamples[test_no])
    print("\nFinal Sorted Score: ")
    print("Rank      Score               Strain Name")
    for idx,(j,k) in enumerate(sorted([(AB_score[i], i) for i in range(total_refStrains)],reverse=True)):
        if idx>14:break        
        print("{:4d}\t{:^13.3f}\t{}".format(idx+1,j,refStrains[k]))

    idx_res =np.argmax(AB_score)
    print("\nSample's Original Class is               : ", testSamples[test_no])
    print("Sample's Predicted Class is              : ", idx_res, refStrains[idx_res])
    print("***************************************************************")
    return test_no, AB_score    


def load_model(model):
    print("\n====== Loading PreTrained Model ======")
    print("MODEL loading from : ",model.input_model_path)
    in_loc = open(model.input_model_path, 'rb')
    global refStrains,total_refStrains,klen,klen_loc,cond_prob,partial_test_interval
    (refStrains,
        total_refStrains,
        klen,
        cond_prob) = pickle.load(in_loc)        
    partial_test_interval=model.partial_test_interval    
    
    print("No of refStrains         : ",total_refStrains)
    print("Value of klen is         : ",klen)
    print("Size of klen dict        : ",len(cond_prob))    
    print("========== DONE ==========\n")


def parse_command(model):
    parser = OptionParser()    
    parser.add_option("--testpath", dest="fastq_testPath",help="Path to folder containing one or more test samples.[default='test_data']",
                    default="test_data")    
    parser.add_option("--output_interval", type="int", dest="partial_test_interval", help="Interval of fastq reads to see predictions.[default=100000] ",default=100000)
    parser.add_option("--cores", type="int", dest="no_of_pools", help="Number of cores to parallelise if multiple test samples.[default=4]", default=4)
    parser.add_option("--input_model" ,dest="input_model_path", help="Location of pretrained model.", default=None)    

    (options, args ) = parser.parse_args() 

    if options.partial_test_interval<0: parser.error("Interval can't be negative")
    if options.no_of_pools > os.cpu_count():parser.error("Your system has only "+str(os.cpu_count())+" cores")  
    if options.no_of_pools <0:parser.error("Invalid number of cores")    
    if not os.path.isdir(options.fastq_testPath):parser.error("Folder does not exist containing test samples.")    
    if len(os.listdir(options.fastq_testPath))==0:parser.error("Empty folder :"+options.fastq_testPath)    
    
    model.fastq_testPath = options.fastq_testPath    
    model.partial_test_interval = int(options.partial_test_interval)
    model.no_of_pools=int(options.no_of_pools)
    model.input_model_path = options.input_model_path 


def test_ibnb():
    global model_test,partial_test_interval
    model_test = IBNB_Model_test()
    parse_command(model_test) 
    partial_test_interval =  model_test.partial_test_interval
    load_model(model_test) 
    model_test.test()   

    
test_ibnb()