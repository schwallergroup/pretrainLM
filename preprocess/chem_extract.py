from chemdataextractor import Document
import pickle
import os

def sort_objects_by_start(object_list):
    return sorted(object_list, key=lambda obj: obj.start)

def extract_entity(path):
    dump_list=[]
    text_file = open("/data/shai/chemistry_thoughts_examples/"+path,"r")
    mmd_text = text_file.read()
    doc = Document(mmd_text)
    sorted_entities = sort_objects_by_start(doc.cems)
    text_file.close()
    for i in sorted_entities:
        dump_list.append([i.text, i.start, i.end])
    with open("/data/shai/chem_thoughts_ent/"+path[:-4]+".pkl","wb") as f:
        pickle.dump(dump_list, f)

if __name__ == "__main__":
    for i in os.listdir("/data/shai/chemistry_thoughts_examples/"):
         extract_entity(i)
