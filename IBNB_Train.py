import os
import sys
import time
import pickle 
import numpy as np
from Bio import SeqIO
from multiprocessing import Pool
from optparse import OptionParser


class IBNB_Model:

    def __init__(self):        
        self.refStrains = []        
        self.refStrains_reads = []                
        self.refStrains_type = "fasta"        
        self.fasta_refPath = []         
        self.klen = 0
        self.hklen = 0
        self.lklen = 0
        self.nucleotides = "ACGT"               
        self.output_model_path= []             

    def readFastaFiles(self):
        for ref in self.refStrains:
            fname = os.path.join(self.fasta_refPath, ref)
            print("Reading Strain :", fname)
            temp=""
            for seq_record in SeqIO.parse(fname, self.refStrains_type):                
                self.refStrains_reads.append(  str(seq_record.seq) )
                break                

    def readRefStrains(self):
        print("========Reading the REFERENCE Strains========")
        self.refStrains = os.listdir(self.fasta_refPath)
        self.refStrains.sort()
        self.total_refStrains = len(self.refStrains)        
        print("Total genomes in Fasta reference folder: ",self.total_refStrains )
        self.readFastaFiles()      
        print("====Reading the REFERENCE Strains Completed====\n")    

    def refStrains_Print(self,start=0,end=0):
        print("==== REFERENCE Strains Detail ====".format(start,end)) 
        for i in range(self.total_refStrains):
            print("Length:- %7d , Strain Name:- %s "%(len(self.refStrains_reads[i]),self.refStrains[i]))

    def fill_hklen_klen_cnt_table(self,reference, ref_idx):
       
        for i in range(len(reference) - self.hklen):            
            hkmer = reference[i:i + self.hklen]
            if hkmer not in self.hklen_cnt_table:
                counts = np.zeros(self.total_refStrains,dtype=np.float32)
                counts[ref_idx] = 1                
                self.hklen_cnt_table[hkmer] = counts
            else:
                counts= self.hklen_cnt_table[hkmer]
                counts[ref_idx] += 1                
                self.hklen_cnt_table[hkmer] = counts
            self.total_hklen_cnt[ref_idx] +=1
                
        for i in range(len(reference) - self.klen):
            kmer = reference[i:i + self.klen]
            if kmer not in self.klen_cnt_table:
                counts = np.zeros(self.total_refStrains,dtype=np.float32)
                counts[ref_idx] = 1                
                self.klen_cnt_table[kmer] = counts
            else:
                counts= self.klen_cnt_table[kmer]
                counts[ref_idx] += 1                
                self.klen_cnt_table[kmer] = counts
            self.total_klen_cnt[ref_idx] +=1

    def fill_cnt_table(self):         
        self.hklen_cnt_table = {}
        self.klen_cnt_table = {}
        self.total_hklen_cnt=np.zeros(self.total_refStrains)
        self.total_klen_cnt=np.zeros(self.total_refStrains)
        print("\n======== Creating the Count Table for Strains ========")
        print("Creating count_table for Strain No : ",end=" ")
        for i in range(self.total_refStrains):
            print(i ,end=" , ")           
            self.fill_hklen_klen_cnt_table(self.refStrains_reads[i] , i)                     
          
        print("\nUnique ",self.klen,"len kmers count   : ",len(self.klen_cnt_table))
        print("Unique ",self.hklen,"hlen hkmers count  : ",len(self.hklen_cnt_table))
        print("\n==== Count Table Created ====\n") 
        

    def fill_cond_prob(self):        
                      
        # print("\n========Filling the Final Weights Table for Strains========")
        # print("Filling cond_prob completed till index : ",end=" ")
        klen_cnt=0
        for kmer in self.klen_cnt_table:
            hkmer=kmer[self.hklen:]            
            counts = np.divide(self.klen_cnt_table[kmer],self.hklen_cnt_table[hkmer],
                                            out=np.zeros(self.total_refStrains,dtype=np.float32),
                                            where = self.hklen_cnt_table[hkmer] !=0 )            
            self.klen_cnt_table[kmer] = counts
            
            klen_cnt+=1
            #if klen_cnt % 200000 == 0:
                #print(klen_cnt, end=" , ")
        # print(klen_cnt,"\n========Filling the Final Weights Table Completed========\n")  

    

    def train(self):
        self.lklen= self.klen-self.hklen             
        self.readRefStrains()
        self.refStrains_Print()
        self.fill_cnt_table()
        self.fill_cond_prob()
        

def parse_command(model):
    parser = OptionParser()
    parser.add_option("--refpath", dest="fasta_refPath", help="Path to folder containing reference fasta sequences.[default='train_data']",default="train_data")    
    parser.add_option("--klen", type="int", dest="klen", help="Length of Kmer to train.(can take values 10 to 12,suggested 10 to 12) [default=12]", default=12)
    parser.add_option("--hklen", type="int", dest="hklen", help="Length of Kmer to train.(max same as klen, suggested half of klen) [default=6]",default=6)    
    parser.add_option("--cores", type="int", dest="no_of_pools", help="Number of cores to parallelise if multiple test samples.[default=4]", default=4)    
    parser.add_option("--output_model" ,dest="output_model_path", help="Location of saving the current training model.", default="IBNB_Model")

    (options, args ) = parser.parse_args() 

    if options.klen<10 or options.klen>12 :parser.error("klen can take values 10 to 12.(Suggested 10 to 12)")
    if options.hklen>options.klen :parser.error("hklen can't be more than klen")    
    if options.no_of_pools > os.cpu_count():parser.error("Your system has only "+str(os.cpu_count())+" cores")  
    if options.no_of_pools <0:parser.error("Invalid number of cores")
    if not os.path.isdir(options.fasta_refPath):parser.error("Folder does not exist containing reference fasta sequences")    
    if len(os.listdir(options.fasta_refPath))==0:parser.error("Empty folder : "+options.fasta_refPath)
    
    model.fasta_refPath = options.fasta_refPath    
    model.klen = int(options.klen)
    model.hklen = int(options.hklen)    
    model.no_of_pools=int(options.no_of_pools)    
    model.output_model_path = options.output_model_path+"_"+str(options.hklen)+"*"+str(options.klen-options.hklen)   
    #return options    


def save_model(model):
    print("\n======= Saving Trained Model =======")
    out_loc = open(model.output_model_path, 'wb')
    pickle.dump((model.refStrains,
                model.total_refStrains,
                model.klen,
                model.klen_cnt_table)    ,out_loc,protocol=4)
    print("MODEL saved at : ",os.path.join(os.getcwd() ,model.output_model_path))  
    print("======== Training Completed =========\n")   
    

def train_ibnb():
    global model_train
    model_train = IBNB_Model()
    parse_command(model_train)
    model_train.train()    
    save_model(model_train)


train_ibnb()
