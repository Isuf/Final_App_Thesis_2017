import os

dataset_keywords = "Experiment"
path = "D:\\Tema NTNU\\Data\\"+dataset_keywords+"\\Deliu\\" 
main_path ="D:\\Tema NTNU\\Data\\Classification"
baseline_path ="D:\\Tema NTNU\\Data\\Delivery"
class Parameters: 

      param = {}
      file_locations={} 
      data_folder=[]
      dataset={}
      baseline_locations={}
      def __init__(self):

          self.baseline_locations={     "hackhound":["hackhound.txt"],
                                        "MovieReview" :["MovieReview_rt-polarity.pos","MovieReview_rt-polarity.neg"] ,
                                        "Subjectivity" :["Subjectivity_plot.tok.gt9.5000","Subjectivity_quote.tok.gt9.5000"],
                                        "Nulled_MC" : ["Nulled\\A_Credentials_2500.txt",   "Nulled\\B_DDOS_500.txt",
                                                       "Nulled\\C_Keyloggers_500.txt",     "Nulled\\D_Crypters_500.txt",
                                                       "Nulled\\E_Trojan_500.txt",         "Nulled\\F_SQLi_500.txt",
                                                       "Nulled\\G_Spam_1500.txt",          "Nulled\\H_RAT_1000.txt",
                                                       "Nulled\\I_NoSecurity_2500.txt"],       

                                        "Nulled_Binary":["Nulled\\NOT_SecurityData_Final.txt","Nulled\\SecurityData_Final.txt"],
                                        "Binary_All":["Nulled\\Binary_All.txt"],
                                        "tmp":["Nulled\\Thesis_Dataset_MultiClass_10K_Final.txt"],
                                        "Nulled_Test":["Nulled\Thesis_Test_Data_1M.txt"]                                   
      
                                        #"LDA_binary":["security_data.txt","irrelevant_data.txt"]
                                  }

          
          self.param={
                      "word2vec" : {
                                    "Train" : True,
                                    "model_name":"w2c_hf_posts.bin",
                                    "vec_size" : 300,
                                    "min_count" : 2,
                                    "Google_w2v" : "D:\\Official_Datasets\\Google word2vec trained\\GoogleNews-vectors-negative300.bin",

                       },

                     "ngrams_bow" : {
                                     "min_ngrams": 3,
                                     "max_ngrams": 3,
                                     "feature_level": "char",   #'char' for character;  'word' for word
                                     "method": "ngrams" ,    # "bow" for Bag-of-Words;  "ngrams" for n(1,2,...) grams  #lsa 
                                     "max_num_features":10000000,
                                     "feature_value": "binary"  # frequency    # TF-IDF  #binary
                       },

                    "ConvNN": { "mode":         "-nonstatic", #-static #-nonstatic
                                "word_vectors": "-word2vec", #rand #word2vec
                                "vector_source" : "ToData", # 1)Google   2)Glove    3)ToData--for train on data   4)Random
                              }
      }
        
          #Creates the full path 
          for i in range(len(self.file_locations)):
               self.file_locations[str(i)] = os.path.join(main_path,  self.file_locations[str(i)])  

          for key, value in self.baseline_locations.items():
              tmp = []
              for loc in value:
                   tmp.append(os.path.join(baseline_path, loc))  
              self.baseline_locations[key]=tmp
 
