import os, sys 
from pathlib import Path
from glob import glob
import tqdm.auto as tqdm
import random
import yaml
import datetime
import shutil

path_to_models = "/is/cluster/fast/scratch/rdanecek/testing/enspark/ablations"
lrs_subset = "pretrain"
# lrs_subset = "test"
video_folder = f"mturk_videos_lrs3/{lrs_subset}"

server_root = "/is/cluster/fast/scratch/rdanecek/testing/enspark/"
path_to_studies = "/is/cluster/fast/scratch/rdanecek/studies/enspark/study_1/"
catch_id = "run0"


def load_catch_videos_lipsync():
    catch_videos_lipsync_correct = []
    catch_videos_lipsync_wrong = []

    path_to_catch_videos = f"/is/cluster/fast/scratch/rdanecek/testing/enspark/catch_trials/{catch_id}/study_1b"
    # find videos that have "true" in their filename
    catch_videos_lipsync_correct = sorted(list(Path(path_to_catch_videos).glob("*true*.mp4")))
    
    # find videos that have "false" in their filename
    catch_videos_lipsync_wrong = sorted(list(Path(path_to_catch_videos).glob("*false*.mp4")))

    return catch_videos_lipsync_correct, catch_videos_lipsync_wrong, Path(path_to_catch_videos)


def load_catch_videos_emo():
    catch_videos_emo_correct = []
    catch_videos_emo_wrong = []

    path_to_catch_videos = f"/is/cluster/fast/scratch/rdanecek/testing/enspark/catch_trials/{catch_id}/study_1a"
    # find videos that have "true" in their filename
    catch_videos_emo_correct = sorted(list(Path(path_to_catch_videos).glob("*true*.mp4")))
    
    # find videos that have "false" in their filename
    catch_videos_emo_wrong = sorted(list(Path(path_to_catch_videos).glob("*false*.mp4")))

    return catch_videos_emo_correct, catch_videos_emo_wrong, Path(path_to_catch_videos)


def find_files(folder, extension):
    # find all files in the folder with the given extension, using linux find 
    files = os.popen("find " + str(folder) + " -name '*." + extension + "'").read().split("\n")
    files = [Path(file) for file in files if file != ""]
    return files


def read_rendered_vids(video_folder): 
    path_to_list = Path(video_folder) / "rendered_list.txt"
    with open(path_to_list, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if lrs_subset in line]
    lines = [Path(line) for line in lines]
    return sorted(lines)


def search_rendered_vids(video_folder): 
    path_to_list = Path(video_folder) / "rendered_list.txt"
    # find all mp4 files in the folder using linux bash find command 
    if not path_to_list.exists():
        cmd = f"find {video_folder} -name '*.mp4'"
        rendered_vids = sorted(os.popen(cmd).read().split("\n")[:-1])
        # dump the list of rendered videos to a file
        with open(path_to_list, "w") as f:
            f.write("\n".join(rendered_vids))
        rendered_vids = [Path(line) for line in rendered_vids]
    else:
        rendered_vids = read_rendered_vids(video_folder)
    return rendered_vids


def replace_rendered_vid_model(video_list, model_path):
    videos_b = []
    for vid in video_list:
        model_idx = 9 
        video_b = Path("/") / "/".join(list(vid.parts[1:model_idx])) / model_path / "/".join(vid.parts[model_idx+1:])
        videos_b += [video_b]
    return videos_b


def check_video_match(videos_a, videos_b):
    # check if the videos match
    parts_to_match = [-2, -3, -4, -5]
    for i in range(len(videos_a)):
        video_a = Path(videos_a[i])
        video_b = Path(videos_b[i])
        for pi in parts_to_match:
            assert video_a.parts[pi] == video_b.parts[pi], f"Videos do not match in part {pi}: {video_a.parts[pi]} vs {video_b.parts[pi]}"
    return


def design_study_1(model_a, model_b, num_rows, num_videos_per_row, output_folder, num_catch_trials=0, num_repeats=5, videos_a=None, videos_b=None):
    model_folder_a = Path(path_to_models) / model_a 
    model_folder_b = Path(path_to_models) / model_b

    video_folder_a = model_folder_a / video_folder
    video_folder_b = model_folder_b / video_folder

    # # get all the videos in the folder
    # videos_a = sorted(list(glob(str(video_folder_a) + "/**/*.mp4", recursive=True)))
    # videos_b = sorted(list(glob(str(video_folder_b) + "/**/*.mp4", recursive=True)))

    # videos_a = find_files(video_folder_a, "mp4")
    # videos_b = find_files(video_folder_b, "mp4")

    videos_a_all = videos_a or read_rendered_vids(video_folder_a)
    videos_b_all = replace_rendered_vid_model(videos_a, model_b)
    # videos_b = read_rendered_vids(video_folder_b)
    
    # remove videos that don't exist in both folders
    random.seed(42)
    video_indices = list(range(len(videos_a)))
    random.shuffle(video_indices)
    videos_a_all = [videos_a_all[i] for i in video_indices]
    videos_b_all = [videos_b_all[i] for i in video_indices]

    # videos_a = videos_a[:100]
    # videos_b = videos_b[:100]

    check_video_match(videos_a_all, videos_b_all)
    assert len(videos_a_all) == len(videos_b_all), "Number of videos in the two folders must be the same"

    # set the random seed

    hit_list_lines = ["images"]
    hit_list_lines_rel = ["images"]
    hit_list_lines_dbg = ["images"]
    
    flips = [[]]
    selected_videos_a_emo = [[]]
    selected_videos_b_emo = [[]]
    selected_videos_a_lip = [[]]
    selected_videos_b_lip = [[]]
    selected_videos_a_relative_to_output = [[]]
    selected_videos_b_relative_to_output = [[]]
    catch_trials = [[]]
    catch_trials_relative_to_output = [[]]
    
    output_folder.mkdir(parents=True, exist_ok=True)

    catch_videos_lip_correct, catch_videos_lip_wrong, lip_catch_path = load_catch_videos_lipsync()
    catch_videos_emo_correct, catch_videos_emo_wrong, emo_catch_path = load_catch_videos_emo()

    # for each video    
    for ri in range(num_rows):
        hit_list_line = []
        hit_list_line_rel = []
        hit_list_line_dbg = []

        video_count = 0
        i = 0

        videos_a = videos_a_all.copy()
        videos_b = videos_b_all.copy()
        while video_count < num_videos_per_row:
            video_a = Path(videos_a[i])
            video_b = Path(videos_b[i])
            # if not video_a.exists() or not video_b.exists():
            #     i += 1
            #     continue

            speaker = video_a.parts[-2].split("_")[0]
            speaker_b = video_b.parts[-2].split("_")[0]
            assert speaker == speaker_b, f"Speakers do not match: {speaker} vs {speaker_b}"
            emotion = video_a.parts[-2].split("_")[1].lower()
            emotion_b = video_b.parts[-2].split("_")[1].lower()
            assert emotion == emotion_b, f"Emotions do not match: {emotion} vs {emotion_b}"
            if emotion.lower() == "neutral":
                i += 1
                continue

            # copy both videos to the output folder (keep the subfolder structure from path_to_models)
            video_a_relative_to_output = Path(video_a).relative_to(path_to_models)
            video_b_relative_to_output = Path(video_b).relative_to(path_to_models)

            video_a_output = output_folder / video_a_relative_to_output
            video_b_output = output_folder / video_b_relative_to_output

            video_a_output.parent.mkdir(parents=True, exist_ok=True)
            video_b_output.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(video_a, video_a_output)
            shutil.copy(video_b, video_b_output)

            # create video_a_output that is mute and video_b_output that is also mute
            video_a_output_mute = video_a_output.parent / (video_a_output.stem + "_mute.mp4")
            video_b_output_mute = video_b_output.parent / (video_b_output.stem + "_mute.mp4")

            # use ffmpeg to mute the videos
            os.system(f"ffmpeg -n -i {video_a_output} -vcodec copy -an {video_a_output_mute}")
            os.system(f"ffmpeg -n -i {video_b_output} -vcodec copy -an {video_b_output_mute}")

            video_a_output_mute_relative_to_output = Path(video_a_output_mute).relative_to(output_folder)
            video_b_output_mute_relative_to_output = Path(video_b_output_mute).relative_to(output_folder)

            # TODO: check if the video names are the same
            

            intensity = video_a.parts[-2].split("_")[2]
            intensity_b = video_b.parts[-2].split("_")[2]
            assert intensity == intensity_b, f"Intensities do not match: {intensity} vs {intensity_b}"

            # randomly flip 50% of the videos
            flip = random.random() > 0.5
            
            flips[ri] += [flip]
            selected_videos_a_emo[ri] += [video_a]
            selected_videos_b_emo[ri] += [video_b]
            selected_videos_a_lip[ri] += [video_a]
            selected_videos_b_lip[ri] += [video_b]

            # hit list line has the following format: videopath_left#videopath_right#emotion#angry;videopath_left#videopath_right#lipsync#happy
            video_a = str(video_a)
            video_b = str(video_b)
            # if debug:
            #     video_a = "http://0.0.0.0:8000/" + str(Path(video_a).relative_to(server_root))
            #     video_b = "http://0.0.0.0:8000/" + str(Path(video_b).relative_to(server_root))
            #     video_a_relative_to_output = "http://0.0.0.0:8000/" + str(Path(video_a_relative_to_output))
            #     video_b_relative_to_output = "http://0.0.0.0:8000/" + str(Path(video_b_relative_to_output))
            #     video_a_output_mute_relative_to_output = "http://0.0.0.0:8000/" + str(Path(video_a_output_mute_relative_to_output))
            #     video_b_output_mute_relative_to_output = "http://0.0.0.0:8000/" + str(Path(video_b_output_mute_relative_to_output))
            video_a_relative_to_output_dbg =  "http://0.0.0.0:8000/" + str(video_a_relative_to_output)
            video_b_relative_to_output_dbg =  "http://0.0.0.0:8000/" + str(video_b_relative_to_output)
            video_a_output_mute_relative_to_output_dbg =   "http://0.0.0.0:8000/" + str(video_a_output_mute_relative_to_output)
            video_b_output_mute_relative_to_output_dbg =   "http://0.0.0.0:8000/" + str(video_b_output_mute_relative_to_output)

            if not flip:
                hit_list_line += [f"{video_a}#{video_b}#emotion#{emotion};{video_a}#{video_b}#lipsync#{emotion}"]
                hit_list_line_rel += [f"{video_a_output_mute_relative_to_output}#{video_b_output_mute_relative_to_output}#emotion#{emotion};{video_a_relative_to_output}#{video_b_relative_to_output}#lipsync#{emotion}"]
                hit_list_line_dbg += [f"{video_a_output_mute_relative_to_output_dbg}#{video_b_output_mute_relative_to_output_dbg}#emotion#{emotion};{video_a_relative_to_output_dbg}#{video_b_relative_to_output_dbg}#lipsync#{emotion}"]
            else:
                hit_list_line += [f"{video_b}#{video_a}#emotion#{emotion};{video_b}#{video_a}#lipsync#{emotion}"]
                hit_list_line_rel += [f"{video_b_output_mute_relative_to_output}#{video_a_output_mute_relative_to_output}#emotion#{emotion};{video_b_relative_to_output}#{video_a_relative_to_output}#lipsync#{emotion}"]
                hit_list_line_dbg += [f"{video_b_output_mute_relative_to_output_dbg}#{video_a_output_mute_relative_to_output_dbg}#emotion#{emotion};{video_b_relative_to_output_dbg}#{video_a_relative_to_output_dbg}#lipsync#{emotion}"]

            # hitlist_str += hit_list_line
            # hitlist_str_rel += hit_list_line_rel
            catch_trials[ri] += [0]
            video_count += 1
            i += 1

        # add catch trials
        for i in range(num_catch_trials):
            flip = random.random() > 0.5
            catch_trials[ri] += [1]
            flips[ri] += [flip]

            random_index = random.randint(0, len(catch_videos_emo_correct)-1)

            video_a_lip = catch_videos_lip_correct[random_index]
            video_b_lip = catch_videos_lip_wrong[random_index]

            video_a_emotion = catch_videos_emo_correct[random_index]
            video_b_emotion = catch_videos_emo_wrong[random_index]

            # video_a_lip_rel = Path(video_a_lip).relative_to(lip_catch_path)
            # video_b_lip_rel = Path(video_b_lip).relative_to(lip_catch_path)
            # video_a_emotion_rel = Path(video_a_emotion).relative_to(emo_catch_path)
            # video_b_emotion_rel = Path(video_b_emotion).relative_to(emo_catch_path)
            
            video_a_lip_out = output_folder / video_a_lip.relative_to(lip_catch_path)
            video_b_lip_out = output_folder / video_b_lip.relative_to(lip_catch_path)
            video_a_emotion_out = output_folder / video_a_emotion.relative_to(emo_catch_path)
            video_b_emotion_out = output_folder / video_b_emotion.relative_to(emo_catch_path)

            shutil.copy(video_a_lip, output_folder / video_a_lip_out)
            shutil.copy(video_b_lip, output_folder / video_b_lip_out)

                    # use ffmpeg to mute the videos
            os.system(f"ffmpeg -n -i {video_a_emotion} -vcodec copy -an {str(output_folder / video_a_emotion_out)}")
            os.system(f"ffmpeg -n -i {video_b_emotion} -vcodec copy -an {str(output_folder / video_b_emotion_out)}")
            # shutil.copy(video_a_emotion, output_folder / video_a_emotion_out)
            # shutil.copy(video_b_emotion, output_folder / video_b_emotion_out)

            video_a_lip_relative_to_output = Path(video_a_lip_out).relative_to(output_folder)
            video_b_lip_relative_to_output = Path(video_b_lip_out).relative_to(output_folder)
            video_a_emotion_relative_to_output = Path(video_a_emotion_out).relative_to(output_folder)
            video_b_emotion_relative_to_output = Path(video_b_emotion_out).relative_to(output_folder)

    
            selected_videos_a_emo[ri] += [video_a_emotion]
            selected_videos_b_emo[ri] += [video_b_emotion]

            selected_videos_a_lip[ri] += [video_a_lip]
            selected_videos_b_lip[ri] += [video_b_lip]

            video_a_lip = str(video_a_lip)
            video_b_lip = str(video_b_lip)
            video_a_emotion = str(video_a_emotion)
            video_b_emotion = str(video_b_emotion)
        
            video_a_lip_relative_to_output = str(video_a_lip_relative_to_output)
            video_b_lip_relative_to_output = str(video_b_lip_relative_to_output)
            video_a_emotion_relative_to_output = str(video_a_emotion_relative_to_output)
            video_b_emotion_relative_to_output = str(video_b_emotion_relative_to_output)

            video_a_lip_dbg = "http://0.0.0.0:8000/" + video_a_lip_relative_to_output
            video_b_lip_dbg = "http://0.0.0.0:8000/" + video_b_lip_relative_to_output
            video_a_emotion_dbg = "http://0.0.0.0:8000/" + video_a_emotion_relative_to_output
            video_b_emotion_dbg = "http://0.0.0.0:8000/" + video_b_emotion_relative_to_output
                
            if not flip:
                hit_list_line += [f"{video_a_emotion}#{video_b_emotion}#emotion#{emotion};{video_a_lip}#{video_b_lip}#lipsync#{emotion}"]
                hit_list_line_rel += [f"{video_a_emotion_relative_to_output}#{video_b_emotion_relative_to_output}#emotion#{emotion};{video_a_lip_relative_to_output}#{video_b_lip_relative_to_output}#lipsync#{emotion}"]
                hit_list_line_dbg += [f"{video_a_emotion_dbg}#{video_b_emotion_dbg}#emotion#{emotion};{video_a_lip_dbg}#{video_b_lip_dbg}#lipsync#{emotion}"]
            else:
                hit_list_line += [f"{video_b_emotion}#{video_a_emotion}#emotion#{emotion};{video_b_lip}#{video_a_lip}#lipsync#{emotion}"]
                hit_list_line_rel += [f"{video_b_emotion_relative_to_output}#{video_a_emotion_relative_to_output}#emotion#{emotion};{video_b_lip_relative_to_output}#{video_a_lip_relative_to_output}#lipsync#{emotion}"]
                hit_list_line_dbg += [f"{video_b_emotion_dbg}#{video_a_emotion_dbg}#emotion#{emotion};{video_b_lip_dbg}#{video_a_lip_dbg}#lipsync#{emotion}"]

        # shuffle to mix in catch trials
        index_list = list(range(len(flips[ri])))
        random.shuffle(index_list)
        flips[ri] = [flips[ri][i] for i in index_list]
        selected_videos_a_emo[ri] = [str(selected_videos_a_emo[ri][i]) for i in index_list]
        selected_videos_b_emo[ri] = [str(selected_videos_b_emo[ri][i]) for i in index_list]
        selected_videos_a_lip[ri] = [str(selected_videos_a_lip[ri][i]) for i in index_list]
        selected_videos_b_lip[ri] = [str(selected_videos_b_lip[ri][i]) for i in index_list]
        catch_trials[ri] = [catch_trials[ri][i] for i in index_list]
        hit_list_line = [hit_list_line[i] for i in index_list]
        hit_list_line_rel = [hit_list_line_rel[i] for i in index_list]
        hit_list_line_dbg = [hit_list_line_dbg[i] for i in index_list]

        # repeat the first num_repeats elements at the end of the list
        flips[ri] += flips[ri][:num_repeats]
        selected_videos_a_emo[ri] += selected_videos_a_emo[ri][:num_repeats]
        selected_videos_b_emo[ri] += selected_videos_b_emo[ri][:num_repeats]
        selected_videos_a_lip[ri] += selected_videos_a_lip[ri][:num_repeats]
        selected_videos_b_lip[ri] += selected_videos_b_lip[ri][:num_repeats]
        catch_trials[ri] += catch_trials[ri][:num_repeats]
        hit_list_line += hit_list_line[:num_repeats]
        hit_list_line_rel += hit_list_line_rel[:num_repeats]
        hit_list_line_dbg += hit_list_line_dbg[:num_repeats]

        hit_list_lines += [';'.join(hit_list_line)]
        hit_list_lines_rel += [';'.join(hit_list_line_rel)]
        hit_list_lines_dbg += [";".join(hit_list_line_dbg)]

        # remove all selected videos from the lists 
        # convert videos to strings
        videos_a_all = [str(video) for video in videos_a_all]
        videos_b_all = [str(video) for video in videos_b_all]

        emo_catch_vids = [str(video) for video in catch_videos_emo_correct + catch_videos_emo_wrong]
        lip_catch_vids = [str(video) for video in catch_videos_lip_correct + catch_videos_lip_wrong]
        

        for rvi in range(len(selected_videos_a_emo[ri])-num_repeats):
            video_to_rm = selected_videos_a_emo[ri][rvi]
            if video_to_rm in emo_catch_vids:
                continue
            videos_a_all.remove(video_to_rm)
        for rvi in range(len(selected_videos_b_emo[ri])-num_repeats):
            video_to_rm = selected_videos_b_emo[ri][rvi]
            if video_to_rm in emo_catch_vids:
                continue
            videos_b_all.remove(video_to_rm)

        # for rvi in range(len(selected_videos_a_lip[ri])-num_repeats):
        #     video_to_rm = selected_videos_a_lip[ri][rvi]
        #     if video_to_rm in lip_catch_vids:
        #         continue
        #     videos_a_all.remove(video_to_rm)
        # for rvi in range(len(selected_videos_b_lip[ri])-num_repeats):
        #     if video_to_rm in lip_catch_vids:
        #         continue
        #     video_to_rm = selected_videos_b_lip[ri][rvi]
        #     videos_b_all.remove(video_to_rm)

        # convert videos back to paths
        videos_a_all = [Path(video) for video in videos_a_all]
        videos_b_all = [Path(video) for video in videos_b_all]

    protocol_dict = {
        "model_a": model_a,
        "model_b": model_b,
        "flips": flips,
        "videos_a_emo": selected_videos_a_emo,
        "videos_b_emo": selected_videos_b_emo, 
        "videos_a_lip": selected_videos_a_lip,
        "videos_b_lip": selected_videos_b_lip,
        "catch_trials": catch_trials,
        "num_repeats": num_repeats,
    }

    protocol_file = Path(output_folder) / "protocol.yaml"
    with open(protocol_file, "w") as f:
        yaml.dump(protocol_dict, f)

    hitlist_file = Path(output_folder) / "hitlist.csv"
    with open(hitlist_file, "w") as f:
        f.write("\n".join(hit_list_lines))
    
    hitlist_rel_file = Path(output_folder) / "hitlist_rel.csv"
    with open(hitlist_rel_file, "w") as f:
        f.write("\n".join(hit_list_lines_rel))

    hitlist_dbg_file = Path(output_folder) / "hitlist_dbg.csv"
    with open(hitlist_dbg_file, "w") as f:
        f.write("\n".join(hit_list_lines_dbg))
    print(f"Protocol file saved to {hitlist_dbg_file}")



def main():
    ## final ENSPARC model (WITH prior, lip reading, video emotion, disentanglement) - to be revised
    main_model = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
    main_model_videos_folder = Path(path_to_models) / main_model / video_folder
    # main_model_videos = read_rendered_vids(main_model_videos_folder)
    main_model_videos = search_rendered_vids(main_model_videos_folder)

    # ablation models
    ablation_models = []
    ## ENSPARC with prior but no perceptual
    ablation_models += ["2023_05_03_22-37-15_3901372200521672564_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"] # lr = 0.0000025, lrd = 0.
    ablation_vids = []

    for model_b in ablation_models:
        vids = replace_rendered_vid_model(main_model_videos, model_b)
        ablation_vids.append(vids)

    # check if all the videos exist 
    indices_to_remove = []
    for vi, vid in enumerate(main_model_videos):
        remove_vid = False
        for ai, abvids in enumerate(ablation_vids):
            if not abvids[vi].exists():
                remove_vid = True
            print(f"Video {vi} of model {ai} exists: {abvids[vi].exists()}")
        if not vid.exists():
            remove_vid = True
        if remove_vid:
            indices_to_remove.append(vi)

    
    for vi in reversed(indices_to_remove):
        for ai, abvids in enumerate(ablation_vids):
            ablation_vids[ai].pop(vi)    
        print("Removing video", vi)
        main_model_videos.pop(vi)

    num_rows = 1
    videos_per_row = 1
    repeats = 0
    num_catch_trials = 3

    # create a timestamped folder for the study
    study_name = "study_"
    using_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    study_name += "_" + using_date + f"nr{num_rows}_r{videos_per_row}_ct{num_catch_trials}_rep{repeats}"

    # dump the video list to a file: 
    video_list_file = Path(path_to_studies) / study_name / "video_list.txt"
    video_list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(video_list_file, "w") as f:
        f.write("\n".join([str(v) for v in main_model_videos]))

    for mi, model_b in enumerate(ablation_models): 
        output_folder = Path(path_to_models) / model_b / "mturk"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_folder = Path(path_to_studies) / study_name / f"main_vs_{mi}"

        design_study_1(main_model, model_b, num_rows, videos_per_row, output_folder, num_catch_trials=num_catch_trials, videos_a=main_model_videos, videos_b=main_model_videos, num_repeats=repeats)
        print("Study folder:", output_folder)

    print("Done design_study_1")


if __name__ == "__main__":
    main()
