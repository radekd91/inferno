import os, sys 
from pathlib import Path
from glob import glob
import tqdm.auto as tqdm
import random
import yaml
import datetime

path_to_models = "/is/cluster/fast/scratch/rdanecek/testing/enspark/ablations"
video_folder = "mturk_videos_lrs3/pretrain"

path_to_studies = "/is/cluster/fast/scratch/rdanecek/studies/enspark/study_3/"

catch_videos_emo = []


def find_files(folder, extension, num_catch_trials=0):
    # find all files in the folder with the given extension, using linux find 
    files = os.popen("find " + str(folder) + " -name '*." + extension + "'").read().split("\n")
    files = [Path(file) for file in files if file != ""]
    return files


def design_study_4(model, real_videos, num_videos, output_folder):
    model_folder = Path(path_to_models) / model 
    real_folder = Path(path_to_models) / real_videos

    model_video_folder = model_folder / video_folder
    real_video_folder = real_folder / video_folder

    # # get all the videos in the folder
    # videos_a = sorted(list(glob(str(video_folder_a) + "/**/*.mp4", recursive=True)))
    # videos_b = sorted(list(glob(str(video_folder_b) + "/**/*.mp4", recursive=True)))

    videos_model = find_files(model_video_folder, "mp4")
    videos_real = find_files(real_video_folder, "mp4")

    # assert len(videos_a) == len(videos_b), "Number of videos in the two folders must be the same"

    # set the random seed
    random.seed(42)

    hitlist_str = "images\n"

    selected_videos_model = []
    selected_videos_real = []
    selected_videos = []
    catch_trials = []

    # for each video    
    for i in range(num_videos):
        video_model = Path(videos_model[i])
        video_real = Path(videos_real[i])

        # TODO: check if the video names are the same

        assert Path(video_model).stem == Path(video_real).stem, "Video names must be the same"
        
        speaker = Path(video_model).stem.split("_")[0]
        emotion = Path(video_model).stem.split("_")[1].lower()
        intensity = Path(video_model).stem.split("_")[2]

        assert emotion == "neutral", "Emotion must be neutral for this study"

        selected_videos_model += [video_model]
        selected_videos_real += [video_real]
        selected_videos += [video_model, video_real]
        catch_trials += [0, 0]

        # hit list line has the following format: videopath_left#videopath_right#emotion#angry;videopath_left#videopath_right#lipsync#happy
        hit_list_line = f"{video_model}\n"
        hitlist_str += hit_list_line
        hit_list_line = f"{video_real}\n"
        hitlist_str += hit_list_line
        

    # add catch trials
    for i in range(num_catch_trials):
        video_catch = random.choice(catch_videos_emo)
        hit_list_line = f"{video_catch}\n"
        catch_trials += [1]

    # shuffle the videos
    index_list = list(range(len(catch_trials)))
    random.shuffle(index_list)

    # selected_videos_model = [selected_videos_model[i] for i in index_list]
    # selected_videos_real = [selected_videos_real[i] for i in index_list]
    catch_trials = [catch_trials[i] for i in index_list]
    selected_videos = [selected_videos[i] for i in index_list]


    protocol_dict = {
        "model": model,
        "real_videos": real_videos,
        "videos_model": selected_videos_model,
        "videos_real": selected_videos_real, 
        "selected_videos": selected_videos, 
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


    for model_b in sota_models: 
        output_folder = Path(path_to_models) / model_b / "mturk"
        output_folder.mkdir(parents=True, exist_ok=True)

        # create a timestamped folder for the study
        study_name = "study_"
        using_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        study_name += "_" + using_date

        output_folder = Path(path_to_studies) / study_name
        output_folder.mkdir(parents=True, exist_ok=True)
        design_study_4(main_model, model_b, 0, output_folder)


if __name__ == "__main__":
    main()
