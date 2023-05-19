import os, sys 
from pathlib import Path
from glob import glob
import tqdm.auto as tqdm
import random
import yaml
import datetime
import shutil

path_to_models = "/is/cluster/fast/scratch/rdanecek/testing/enspark/ablations"
path_to_baslines = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/"
# lrs_subset = "pretrain"
lrs_subset = "test"
video_folder = f"mturk_videos_lrs3/{lrs_subset}"

server_root = "/is/cluster/fast/scratch/rdanecek/testing/enspark/"
path_to_studies = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/study_2/"
catch_id = "run18"

bucket_prefix = "https://ensparc.s3.eu-central-1.amazonaws.com/"


def load_catch_videos_study_2():
    path_to_catch_videos = f"/is/cluster/fast/scratch/rdanecek/testing/enspark/catch_trials/{catch_id}/study_2"
    # find videos that have "true" in their filename
    catch_videos = sorted(list(Path(path_to_catch_videos).glob("*true*.mp4")))
    correct_answers = [Path(video).stem.split("_")[2] for video in catch_videos]
    # filter out videos that do not have the correct identifier
    return catch_videos, correct_answers, Path(path_to_catch_videos)


def find_files(folder, extension):
    # find all files in the folder with the given extension, using linux find 
    files = os.popen("find " + str(folder) + " -name '*." + extension + "'").read().split("\n")
    files = [Path(file) for file in files if file != ""]
    return files


def read_rendered_vids_all(video_folder): 
    path_to_list = Path(video_folder) / "rendered_list_all.txt"
    with open(path_to_list, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if lrs_subset in line]
    lines = [Path(line) for line in lines]
    return sorted(lines)


def search_all_rendered_vids(video_folder): 
    path_to_list = Path(video_folder) / "rendered_list_all.txt"
    # find all mp4 files in the folder using linux bash find command, they must inlucde "Neutral" in full path
    if not path_to_list.exists():
        cmd = f"find {video_folder} -name '*.mp4'" 
        rendered_vids = sorted(os.popen(cmd).read().split("\n")[:-1])
        # dump the list of rendered videos to a file
        with open(path_to_list, "w") as f:
            f.write("\n".join(rendered_vids))
        rendered_vids = [Path(line) for line in rendered_vids]
    else:
        rendered_vids = read_rendered_vids_all(video_folder)
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



def design_study_2(model, num_rows, num_videos_per_row, output_folder, num_catch_trials=0, num_repeats=5, videos=None, videos_b=None):
    model_folder_a = Path(path_to_models) / model 
    # model_folder_b = Path(path_to_models) / model_b

    video_folder_a = model_folder_a / video_folder
    # video_folder_b = model_folder_b / video_folder

    # # get all the videos in the folder
    # videos_a = sorted(list(glob(str(video_folder_a) + "/**/*.mp4", recursive=True)))
    # videos_b = sorted(list(glob(str(video_folder_b) + "/**/*.mp4", recursive=True)))

    # videos_a = find_files(video_folder_a, "mp4")
    # videos_b = find_files(video_folder_b, "mp4")

    videos_all = videos or read_rendered_vids_all(video_folder_a)
    # videos_b_all = replace_rendered_vid_model(videos_a, model_b)
    # videos_b = read_rendered_vids(video_folder_b)
    
    # remove videos that don't exist in both folders
    random.seed(42)
    video_indices = list(range(len(videos)))
    random.shuffle(video_indices)
    videos_all = [videos_all[i] for i in video_indices]
    # videos_b_all = [videos_b_all[i] for i in video_indices]

    

    # check_video_match(videos_a_all, videos_b_all)
    # assert len(videos_a_all) == len(videos_b_all), "Number of videos in the two folders must be the same"

    # set the random seed

    hit_list_lines = ["images"]
    hit_list_lines_rel = ["images"]
    hit_list_lines_dbg = ["images"]
    hit_list_lines_mturk = ["images"]
    
    flips = [[]]
    selected_videos = [[]]
    correct_emotions = [[]]
    catch_trials = [[]]
    
    output_folder.mkdir(parents=True, exist_ok=True)
    study_folder = output_folder.parent.parent.parent

    catch_videos, catch_answers, catch_path = load_catch_videos_study_2()
    
    # for each video    
    for ri in range(num_rows):
        hit_list_line = []
        hit_list_line_rel = []
        hit_list_line_dbg = []
        hit_list_line_mturk = []

        video_count = 0
        i = 0

        videos = videos_all.copy()
        # videos_b = videos_b_all.copy()
        while video_count < num_videos_per_row:
            video = Path(videos[i])
            # video_b = Path(videos_b[i])
            # if not video_a.exists() or not video_b.exists():
            #     i += 1
            #     continue

            emotion = video.parts[-2].split("_")[1].lower()
            # assert emotion.lower() == "neutral", f"Emotion must be neutral: {emotion}"

            # copy both videos to the output folder (keep the subfolder structure from path_to_models)
            video_relative_to_output = Path(video).relative_to(path_to_models)
            # video_b_relative_to_output = Path(video_b).relative_to(path_to_baslines)

            video_output = output_folder / video_relative_to_output
            # video_b_output = output_folder / video_b_relative_to_output

            video_output.parent.mkdir(parents=True, exist_ok=True)

            video_relative_to_output = Path(video_output).relative_to(study_folder)

            # # create video_a_output that is mute and video_b_output that is also mute
            video_output_mute = video_output.parent / (video_output.stem + "_mute.mp4")
            video_relative_to_output_mute  = Path(video_output_mute).relative_to(study_folder)

    
            # # use ffmpeg to mute the videos
            video_output_mute.parent.mkdir(parents=True, exist_ok=True)
            os.system(f"ffmpeg -n -i {video} -vcodec copy -an {video_output_mute}")

            selected_videos[ri] += [video]
            correct_emotions[ri] += [emotion]

            # hit list line has the following format: videopath_left#videopath_right#emotion#angry;videopath_left#videopath_right#lipsync#happy
            video = str(video)

            video_relative_to_output_dbg =  "http://0.0.0.0:8000/" + str(video_relative_to_output)
            video_relative_to_output_mute_dbg =  "http://0.0.0.0:8000/" + str(video_relative_to_output_mute)
            video_mturk = bucket_prefix + str(video_relative_to_output)
            video_mturk_mute = bucket_prefix + str(video_relative_to_output_mute)

            hit_list_line += [f"{video}"]
            hit_list_line_rel += [f"{video_relative_to_output_mute}"]
            hit_list_line_dbg += [f"{video_relative_to_output_mute_dbg}"]
            hit_list_line_mturk += [f"{video_mturk_mute}"]

            catch_trials[ri] += [0]
            video_count += 1
            i += 1

        # add catch trials
        for i in range(num_catch_trials):
            catch_trials[ri] += [1]

            random_index = random.randint(0, len(catch_videos)-1)

            video = catch_videos[random_index]
            answer = catch_answers[random_index]
            
            out_video = output_folder / video.relative_to(catch_path)
            os.system(f"ffmpeg -n -i {video} -vcodec copy -an {str(out_video)}")
            # shutil.copy(video, output_folder / video)

            video_relative_to_output = Path(out_video).relative_to(study_folder)

            selected_videos[ri] += [video]
            correct_emotions[ri] += [answer]

            video = str(video)
        
            video_relative_to_output = str(video_relative_to_output)

            video_a_lip_dbg = "http://0.0.0.0:8000/" + video_relative_to_output

            video_a_lip_mturk = bucket_prefix + str(video_relative_to_output)
            # if not flip:
            hit_list_line += [f"{video}"]
            hit_list_line_rel += [f"{video_relative_to_output}"]
            hit_list_line_dbg += [f"{video_a_lip_dbg}"]
            hit_list_line_mturk += [f"{video_a_lip_mturk}"]


        assert len(selected_videos[ri]) == len(catch_trials[ri]) 
        assert len(correct_emotions[ri]) == len(catch_trials[ri])

        # shuffle to mix in catch trials
        index_list = list(range(len(selected_videos[ri])))
        random.shuffle(index_list)
        # flips[ri] = [flips[ri][i] for i in index_list]
        selected_videos[ri] = [str(selected_videos[ri][i]) for i in index_list]
        catch_trials[ri] = [catch_trials[ri][i] for i in index_list]
        hit_list_line = [hit_list_line[i] for i in index_list]
        hit_list_line_rel = [hit_list_line_rel[i] for i in index_list]
        hit_list_line_dbg = [hit_list_line_dbg[i] for i in index_list]
        hit_list_line_mturk = [hit_list_line_mturk[i] for i in index_list]

        # repeat the first num_repeats elements at the end of the list
        # flips[ri] += flips[ri][:num_repeats]
        selected_videos[ri] += selected_videos[ri][:num_repeats]
        catch_trials[ri] += catch_trials[ri][:num_repeats]
        hit_list_line += hit_list_line[:num_repeats]
        hit_list_line_rel += hit_list_line_rel[:num_repeats]
        hit_list_line_dbg += hit_list_line_dbg[:num_repeats]
        hit_list_line_mturk += hit_list_line_mturk[:num_repeats]

        hit_list_lines += [';'.join(hit_list_line)]
        hit_list_lines_rel += [';'.join(hit_list_line_rel)]
        hit_list_lines_dbg += [";".join(hit_list_line_dbg)]
        hit_list_lines_mturk += [";".join(hit_list_line_mturk)]

        # remove all selected videos from the lists 
        # convert videos to strings
        videos_all = [str(video) for video in videos_all]

        catch_vids_str = [str(video) for video in catch_videos]
        
        for rvi in range(len(selected_videos[ri])-num_repeats):
            video_to_rm = selected_videos[ri][rvi]
            if video_to_rm in catch_vids_str:
                continue
            videos_all.remove(video_to_rm)

        # convert videos back to paths
        videos_all = [Path(video) for video in videos_all]
        # videos_b_all = [Path(video) for video in videos_b_all]

    assert len(selected_videos) == len(catch_trials) 
    assert len(correct_emotions) == len(catch_trials)

    protocol_dict = {
        "model": model,
        "videos": selected_videos,
        "catch_trials": catch_trials,
        "num_repeats": num_repeats,
        "num_catch_trials": num_catch_trials,
        "correct_emotions": correct_emotions,
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

    hitlist_mturk_file = Path(output_folder) / "hitlist_mturk.csv"
    with open(hitlist_mturk_file, "w") as f:
        f.write("\n".join(hit_list_lines_mturk))
    print(f"Protocol file saved to {hitlist_dbg_file}")
    return hit_list_lines_mturk



def main():
    main_model = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
    main_model_videos_folder = Path(path_to_models) / main_model / video_folder
    # main_model_videos = read_rendered_vids(main_model_videos_folder)
    main_model_videos = search_all_rendered_vids(main_model_videos_folder)

    # ablation models
    ablation_models = [main_model]
    ## ENSPARC with prior but no perceptual
    ablation_models += ["2023_05_03_22-37-15_3901372200521672564_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_predEJ_LVm"] # lr = 0.0000025, lrd = 0.
    ablation_vids = []

    for model in ablation_models:
        vids = replace_rendered_vid_model(main_model_videos, model)
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
    videos_per_row = 35
    repeats = 5
    num_catch_trials = 3

    # create a timestamped folder for the study
    study_name = "study_"
    using_date = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    study_name += "_" + using_date + f"__nr{num_rows}_r{videos_per_row}_ct{num_catch_trials}_rep{repeats}"

    # dump the video list to a file: 
    video_list_file = Path(path_to_studies) / study_name / "video_list.txt"
    video_list_file.parent.mkdir(parents=True, exist_ok=True)
    with open(video_list_file, "w") as f:
        f.write("\n".join([str(v) for v in main_model_videos]))

    hitlists = []
    for mi, model in enumerate(ablation_models): 
        output_folder = Path(path_to_models) / model / "mturk"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_folder = Path(path_to_studies) / study_name / f"{mi}"

        model_vids = ablation_vids[mi]
        hitlist = design_study_2(model, num_rows, videos_per_row, output_folder, num_catch_trials=num_catch_trials, videos=model_vids, num_repeats=repeats)
        hitlists.append(hitlist)
        print("Study folder:", output_folder)

    
    hitlist_all = ["images"]
    for mi, hitlist in enumerate(hitlists):
        hitlist_all += hitlist[1:]

    hitlist_all_file = Path(path_to_studies) / study_name / "hitlist_all.csv"
    with open(hitlist_all_file, "w") as f:
        f.write("\n".join(hitlist_all))

    print("Done design_study_3")


if __name__ == "__main__":
    main()
