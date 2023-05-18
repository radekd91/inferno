import os, sys 
from pathlib import Path
from glob import glob
import tqdm.auto as tqdm
import random
import yaml
import datetime

path_to_models = "/is/cluster/fast/scratch/rdanecek/testing/enspark/ablations"
video_folder = "mturk_videos_lrs3/pretrain"

path_to_studies = "/is/cluster/fast/scratch/rdanecek/studies/enspark/study_1/"


catch_videos_lipsync = []
catch_videos_emo = []


def find_files(folder, extension, num_catch_trials=0):
    # find all files in the folder with the given extension, using linux find 
    files = os.popen("find " + str(folder) + " -name '*." + extension + "'").read().split("\n")
    files = [Path(file) for file in files if file != ""]
    return files


def design_study_1(model_a, model_b, num_videos, output_folder):
    model_folder_a = Path(path_to_models) / model_a 
    model_folder_b = Path(path_to_models) / model_b

    video_folder_a = model_folder_a / video_folder
    video_folder_b = model_folder_b / video_folder

    # # get all the videos in the folder
    # videos_a = sorted(list(glob(str(video_folder_a) + "/**/*.mp4", recursive=True)))
    # videos_b = sorted(list(glob(str(video_folder_b) + "/**/*.mp4", recursive=True)))

    videos_a = find_files(video_folder_a, "mp4")
    videos_b = find_files(video_folder_b, "mp4")

    # assert len(videos_a) == len(videos_b), "Number of videos in the two folders must be the same"

    # set the random seed
    random.seed(42)

    hitlist_str = "images\n"

    flips = []
    selected_videos_a = []
    selected_videos_b = []
    catch_trials = []

    # for each video    
    for i in range(num_videos):
        video_a = Path(videos_a[i])
        video_b = Path(videos_b[i])

        # TODO: check if the video names are the same

        assert Path(video_a).stem == Path(video_b).stem, "Video names must be the same"
        
        speaker = Path(video_a).stem.split("_")[0]
        emotion = Path(video_a).stem.split("_")[1].lower()
        intensity = Path(video_a).stem.split("_")[2]

        # randomly flip 50% of the videos
        flip = random.random() > 0.5
        
        flips += [flip]
        selected_videos_a += [video_a]
        selected_videos_b += [video_b]

        # hit list line has the following format: videopath_left#videopath_right#emotion#angry;videopath_left#videopath_right#lipsync#happy

        if not flip:
            hit_list_line = f"{video_a}#{video_b}#emotion#{emotion};{video_a}#{video_b}#lipsync#{emotion}\n"
        else:
            hit_list_line = f"{video_b}#{video_a}#emotion#{emotion};{video_b}#{video_a}#lipsync#{emotion}\n"
        hitlist_str += hit_list_line
        catch_trials += [0]

    # add catch trials
    for i in range(num_catch_trials):
        flip = random.random() > 0.5
        catch_trials += [1]
        flips += [flip]
        video_a = random.choice(videos_a)
        video_b_lip = random.choice(catch_videos_lipsync)
        video_b_emotion =  random.choice(catch_videos_emo)
        if not flip:
            hit_list_line = f"{video_a}#{video_b_emotion}#emotion#{emotion};{video_a}#{video_b_lip}#lipsync#{emotion}\n"
        else:
            hit_list_line = f"{video_b_emotion}#{video_a}#emotion#{emotion};{video_b_lip}#{video_a}#lipsync#{emotion}\n"

    index_list = list(range(len(flips)))
    random.shuffle(index_list)

    flips = [flips[i] for i in index_list]
    selected_videos_a = [selected_videos_a[i] for i in index_list]
    selected_videos_b = [selected_videos_b[i] for i in index_list]
    catch_trials = [catch_trials[i] for i in index_list]


    protocol_dict = {
        "model_a": model_a,
        "model_b": model_b,
        "flips": flips,
        "videos_a": selected_videos_a,
        "videos_b": selected_videos_b, 
        "catch_trials": catch_trials,
    }

    protocol_file = Path(output_folder) / "protocol.yaml"
    with open(protocol_file, "w") as f:
        yaml.dump(protocol_dict, f)

    hitlist_file = Path(output_folder) / "hitlist.csv"
    with open(hitlist_file, "w") as f:
        f.write(hitlist_str)

    print(f"Protocol file saved to {protocol_file}")



def main():
    main_model = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"

    # ablation models
    ablation_models = []
    ## ENSPARC - perceptual (WITH prior,  WITHOUT lip reading, video emotion, disentanglement) - to be revised
    ablation_models += ["2023_05_12_15-03-26_-1019690419220233481_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"] # lr = 0.0000025, lrd = 0.

    for model_b in ablation_models: 
        output_folder = Path(path_to_models) / model_b / "mturk"
        output_folder.mkdir(parents=True, exist_ok=True)

        # create a timestamped folder for the study
        study_name = "study_"
        using_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        study_name += "_" + using_date

        output_folder = Path(path_to_studies) / study_name
        output_folder.mkdir(parents=True, exist_ok=True)
        design_study_1(main_model, model_b, 0, output_folder)


if __name__ == "__main__":
    main()
