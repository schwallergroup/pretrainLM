from pubchempy import PubChemHTTPError, get_compounds
from py2opsin import py2opsin
from rdkit import Chem
import pubchempy
import time
import os
import urllib
import pickle
import ollama
import numpy as np
import json

import re

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
import pickle

#import multiprocessing as mp
#from multiprocessing import Pool

class SentenceTransformerClassifier(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes=4):
        super(SentenceTransformerClassifier, self).__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token representation
        logits = self.classifier(pooled_output)
        probabilities = self.softmax(logits)
        return probabilities

    def save_model(self, path):
        torch.save(self.state_dict(), path)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path, model_name='sentence-transformers/all-MiniLM-L6-v2', num_classes=4):
        model = cls(model_name, num_classes)
        model.load_state_dict(torch.load(path))
        model.eval()
        print(f"Model loaded from {path}")
        return model
            

def extract_category(text):
    pattern = r"Category:\s*(\d+(?:\.\d+)?)"
    match = re.search(pattern, text)
    
    if match:
        score = float(match.group(1))
        return score
    else:
        return None

def canonicalize(smiles):
    mol_obj = Chem.MolFromSmiles(smiles)
    if mol_obj is not None:
        clean_smiles = Chem.MolToSmiles(mol_obj)
    else:
        clean_smiles = None
    return clean_smiles

smiles_dict = {symbol: f'[{symbol}]' for symbol in 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og'.split()}
empty = []


model = SentenceTransformerClassifier().to("cuda")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', device_map="cuda")
model.load_state_dict(torch.load("/data/shai/weights_new_large/sentence_classifier_model74.pth"))

def annotate(path, smiles_dict, empty):
    text_file = open("/data/shai/chemistry_thoughts_examples/"+path,"r")

    list_file = open("/data/shai/chem_thoughts_ent/"+path[:-4]+".pkl","rb")
    sorted_entities = pickle.load(list_file)
    list_file.close()

    #binary_file = open("/data/shai/test_chem/10.26434_chemrxiv-2021-0d01d-v2.mmd","rb")
    #text_file = open("/data/shai/test_chem/10.26434_chemrxiv-2021-0d01d-v2.mmd","r")

    new_text = ""
    text = text_file.read()

    offset = 0
    #print(sorted_entities)
    total_entity = len(sorted_entities)
    i=0
    while i < len(sorted_entities):
        score = None
        entity = sorted_entities[i]
        entity_text = entity[0]
        entity_text = entity_text.strip(" \n").strip(",")
        smiles = ""
        insert_text = " ".join(text[entity[1]-30:entity[1]].split(" ")[1:]) +"<ENTITY>"+entity_text+"</ENTITY> "+ " ".join(text[entity[2]:entity[2]+20].split(" ")[1:])
        #print(insert_text)
        #print(smiles_dict)
        if entity_text in smiles_dict.keys():
            smiles = smiles_dict[entity_text]
        elif entity_text in empty:
            i+=1
            continue
        else:
            smiles = ""
            new_input = tokenizer(insert_text, padding=True, truncation=True, return_tensors="pt").to("cuda")
            model.eval()
            with torch.no_grad():
                #print(new_input)
                new_prob = model(new_input['input_ids'].to("cuda"), new_input['attention_mask'].to("cuda"))
                _, new_pred = torch.max(new_prob, dim=1)
            score = int(new_pred.item())
            #print(score)
            if score == 0:
                mol_formula = py2opsin(entity_text)
                if len(mol_formula) != "":
                        smiles = canonicalize(mol_formula)
                        if smiles is None:
                            empty.append(entity_text)
                            i+=1
                            continue
                        smiles_dict[entity_text] = smiles
            elif score == 1:
                try:
                    mol_formula = get_compounds(entity_text,"formula")
                    if len(mol_formula) > 0:
                        smiles = canonicalize(mol_formula[0].isomeric_smiles)
                        if smiles is None:
                            i+=1
                            empty.append(entity_text)
                            continue
                        smiles_dict[entity_text] = smiles
                        #new_text += text[offset:entity[1]]
                        #new_text+="\ce{"
                        #new_text+=text[entity[1]:entity[2]]
                        #new_text+="}"
                        #offset = entity[2]
                        #i+=1
                        #continue
                    #mol_formula = get_compounds(entity_text,"formula")
                    #if len(mol_formula) > 0:
                    #        smiles = canonicalize(mol_formula[0].isomeric_smiles)
                    #        if smiles is None:
                    #            i+=1
                    #            empty.append(entity_text)
                    #            continue
                    #        smiles_dict[entity_text] = smiles
                except pubchempy.PubChemHTTPError:
                    empty.append(entity_text)
                    i+=1
                    continue
                except pubchempy.BadRequestError:
                    empty.append(entity_text)
                    i+=1
                    continue
                except pubchempy.TimeoutError:
                    time.sleep(0.2)
                    continue
                except pubchempy.ServerError:
                    i+=1
                    continue
                except urllib.error.URLError:
                    time.sleep(0.2)
                    continue
                except json.decoder.JSONDecodeError:
                    empty.append(entity_text)
                    i+=1
                    continue
                except ValueError:
                    empty.append(entity_text)
                    i+=1
                    continue
            elif score == 2:
                mol_ids = []
                try:
                    mol_ids = get_compounds(entity_text, "name")
                except pubchempy.PubChemHTTPError:
                    empty.append(entity_text)
                    i+=1
                    continue
                except pubchempy.ServerError:
                    empty.append(entity_text)
                    i+=1
                    continue
                except pubchempy.TimeoutError:
                    time.sleep(0.2)
                    continue
                except urllib.error.URLError:
                    time.sleep(0.2)
                    continue
                except json.decoder.JSONDecodeError:
                    i+=1
                    continue
                except ValueError:
                    empty.append(entity_text)
                    i+=1
                    continue
                if len(mol_ids)>0:
                    smiles = canonicalize(mol_ids[0].isomeric_smiles)
                    if smiles is None:
                        i+=1
                        continue
                    smiles_dict[entity_text] = smiles
                else:
                    empty.append(entity_text)
                    i+=1
                    continue
            elif score == 3:
                empty.append(entity_text)
                i+=1
                continue
        if smiles.strip(" ") not in ["" , None]:
            new_text += text[offset:entity[1]]
            new_text += "[START_MOL] "
            new_text += text[entity[1]:entity[2]]
            new_text += " [END_MOL]"
            new_text += "[START_SMILES] "
            new_text += smiles
            new_text +=  " [END_SMILES]"
            offset = entity[2]
        i+=1
        continue
        
    if offset!=len(text):
        new_text += text[offset:len(text)]

    write_file = open("/data/shai/chem_thoughts_smiles/"+path,"w")
    write_file.write(new_text)
    write_file.close()
    return smiles_dict, empty




if __name__ == "__main__":
    count = 0
    #empty = []
    g = open("/data/shai/empty4.pkl","rb")
    f = open("/home/shai/string2smiles3.pkl","rb")
    smiles_dict = pickle.load(f)
    empty = pickle.load(g)
    g.close()
    f.close()
    #pool = mp.Pool(processes=4)
    dir_list = os.listdir("/data/shai/chemistry_thoughts_examples/")
    for i in dir_list[3378:]:
        smiles_dict, empty = annotate(i, smiles_dict, empty)
        count+=1
        if count%10==0:
            g = open("/data/shai/empty4.pkl","wb")
            f = open("/home/shai/string2smiles3.pkl","wb")
            pickle.dump(smiles_dict, f)
            pickle.dump(empty, g)
            g.close()
            f.close()
