import os, sys 
from pathlib import Path
from tqdm import auto


def main(): 
    folder = Path(sys.argv[1])

    # for a given folder, find all subfolders 
    subfolders = sorted([f for f in folder.iterdir() if f.is_dir()])
    # sort them 
    
    num_to_keep = 50
    subfolders = subfolders[num_to_keep:]
    
    # delete all the subfolders
    for subfolder in auto.tqdm(subfolders):
        os.system(f"rm -rf {str(subfolder)}")

    print("Done")
        





if __name__ == "__main__": 
    main()