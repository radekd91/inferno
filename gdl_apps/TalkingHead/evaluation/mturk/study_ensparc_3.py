import os, sys 
from pathlib import Path
from glob import glob
import tqdm.auto as tqdm
import random
import yaml
import datetime
import shutil

# path_to_models = "/is/cluster/fast/scratch/rdanecek/testing/enspark/ablations"
path_to_models = "/is/cluster/work/rdanecek/testing/enspark/ablations"
path_to_baslines = "/is/cluster/fast/scratch/rdanecek/testing/enspark/baselines/"
# lrs_subset = "pretrain"
lrs_subset = "test"
video_folder = f"mturk_videos_lrs3/{lrs_subset}"

server_root = "/is/cluster/fast/scratch/rdanecek/testing/enspark/"
# path_to_studies = "/is/cluster/fast/scratch/rdanecek/studies/enspark_v2/study_3/"
# catch_id = "run18"
path_to_studies = "/is/cluster/fast/scratch/rdanecek/studies/enspark_final_v0/study_3/"
catch_id = "run11"

bucket_prefix = "https://ensparc.s3.eu-central-1.amazonaws.com/"


def load_catch_videos_lipsync(method):
    catch_videos_lipsync_correct = []
    catch_videos_lipsync_wrong = []

    identifier = method.split("_")[0]

    if identifier not in ["CT", "VOCA", "MT", "FF"]: 
        identifier = "FF"
        print("WARNING: identifier not recognized, using FF as default for method: ", method)


    # path_to_catch_videos = f"/is/cluster/fast/scratch/rdanecek/testing/enspark/catch_trials/{catch_id}/study_3"
    path_to_catch_videos = f"/is/cluster/fast/scratch/rdanecek/testing/enspark/catch_trials/new_renders/{catch_id}/study_3"

    # find videos that have "true" in their filename
    catch_videos_lipsync_correct = sorted(list(Path(path_to_catch_videos).glob("*true*.mp4")))
    # filter out videos that do not have the correct identifier
    catch_videos_lipsync_correct = [video for video in catch_videos_lipsync_correct if identifier in video.stem]

    # find videos that have "false" in their filename
    catch_videos_lipsync_wrong = sorted(list(Path(path_to_catch_videos).glob("*false*.mp4")))
    # filter out videos that do not have the correct identifier
    catch_videos_lipsync_wrong = [video for video in catch_videos_lipsync_wrong if identifier in video.stem]

    return catch_videos_lipsync_correct, catch_videos_lipsync_wrong, Path(path_to_catch_videos)



def find_files(folder, extension):
    # find all files in the folder with the given extension, using linux find 
    files = os.popen("find " + str(folder) + " -name '*." + extension + "'").read().split("\n")
    files = [Path(file) for file in files if file != ""]
    return files


def read_rendered_vids_neutral(video_folder): 
    path_to_list = Path(video_folder) / "rendered_list_neutral.txt"
    with open(path_to_list, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if lrs_subset in line]
    lines = [Path(line) for line in lines]
    return sorted(lines)


def search_neutral_rendered_vids(video_folder, pattern = None): 
    path_to_list = Path(video_folder) / "rendered_list_neutral.txt"
    # find all mp4 files in the folder using linux bash find command, they must inlucde "Neutral" in full path
    if not path_to_list.exists():
        cmd = f"find {video_folder} -name '*.mp4' | grep Neutral_0" 
        rendered_vids = sorted(os.popen(cmd).read().split("\n")[:-1])
        if pattern is not None:
            rendered_vids = [vid for vid in rendered_vids if pattern in vid]
        # dump the list of rendered videos to a file
        with open(path_to_list, "w") as f:
            f.write("\n".join(rendered_vids))
        rendered_vids = [Path(line) for line in rendered_vids]
    else:
        rendered_vids = read_rendered_vids_neutral(video_folder)
    return rendered_vids


def replace_rendered_baseline_model(video_list, model_path):
    videos_b = []
    for vid in video_list:
        video_b = Path(path_to_baslines) / model_path / vid.parts[11] / vid.parts[12] / vid.name
        videos_b += [video_b]
    return videos_b

def replace_rendered_vid_model(video_list, model_path, model_idx=9):
    videos_b = []
    for vid in video_list:
        video_b = Path("/") / "/".join(list(vid.parts[1:model_idx])) / model_path / "/".join(vid.parts[model_idx+1:])
        videos_b += [video_b]
    return videos_b


def check_video_match(videos_a, videos_b):
    # check if the videos match
    # parts_to_match = [-2, -3, -4, -5]
    for i in range(len(videos_a)):
        video_a = Path(videos_a[i])
        video_b = Path(videos_b[i])
        # for pi in parts_to_match:
        assert video_a.parts[-4:-2] == video_b.parts[-3:-1], f"Videos do not match in filenames {video_a} {video_b}"
    return



def check_video_match_ablation(videos_a, videos_b):
    # check if the videos match
    # parts_to_match = [-2, -3, -4, -5]
    for i in range(len(videos_a)):
        video_a = Path(videos_a[i])
        video_b = Path(videos_b[i])
        # for pi in parts_to_match:
        assert video_a.parts[-4:-1] == video_b.parts[-4:-1], f"Videos do not match in filenames {video_a} {video_b}"
    return



def design_study_3(model_a, model_b, num_rows, num_videos_per_row, output_folder, num_catch_trials=0, num_repeats=5, 
                   videos_a=None, is_baseline=True):
    model_folder_a = Path(path_to_models) / model_a 
    model_folder_b = Path(path_to_models) / model_b

    video_folder_a = model_folder_a / video_folder
    video_folder_b = model_folder_b / video_folder

    # # get all the videos in the folder
    # videos_a = sorted(list(glob(str(video_folder_a) + "/**/*.mp4", recursive=True)))
    # videos_b = sorted(list(glob(str(video_folder_b) + "/**/*.mp4", recursive=True)))

    # videos_a = find_files(video_folder_a, "mp4")
    # videos_b = find_files(video_folder_b, "mp4")

    videos_a_all = videos_a or read_rendered_vids_neutral(video_folder_a)
    if is_baseline:
        videos_b_all = replace_rendered_baseline_model(videos_a, model_b)
    else:
        videos_b_all = replace_rendered_vid_model(videos_a, model_b, model_idx=8)
    # videos_b = read_rendered_vids(video_folder_b)
    
    # remove videos that don't exist in both folders
    random.seed(42)
    video_indices = list(range(len(videos_a)))
    random.shuffle(video_indices)
    videos_a_all = [videos_a_all[i] for i in video_indices]
    videos_b_all = [videos_b_all[i] for i in video_indices]

    # videos_a = videos_a[:100]
    # videos_b = videos_b[:100]

    if is_baseline:
        check_video_match(videos_a_all, videos_b_all)
    else:
        check_video_match_ablation(videos_a_all, videos_b_all)
    assert len(videos_a_all) == len(videos_b_all), "Number of videos in the two folders must be the same"

    # set the random seed

    hit_list_lines = ["images"]
    hit_list_lines_rel = ["images"]
    hit_list_lines_dbg = ["images"]
    hit_list_lines_mturk = ["images"]
    
    flips = [[]]
    selected_videos_a_lip = [[]]
    selected_videos_b_lip = [[]]
    catch_trials = [[]]
    
    output_folder.mkdir(parents=True, exist_ok=True)
    study_folder = output_folder.parent.parent.parent

    catch_videos_lip_correct, catch_videos_lip_wrong, lip_catch_path = load_catch_videos_lipsync(model_b)
    
    # for each video    
    for ri in range(num_rows):
        hit_list_line = []
        hit_list_line_rel = []
        hit_list_line_dbg = []
        hit_list_line_mturk = []

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

            # speaker = video_a.parts[-2].split("_")[0]
            # speaker_b = video_b.parts[-2].split("_")[0]
            # assert speaker == speaker_b, f"Speakers do not match: {speaker} vs {speaker_b}"
            emotion = video_a.parts[-2].split("_")[1].lower()
            assert emotion.lower() == "neutral", f"Emotion must be neutral: {emotion}"

            # copy both videos to the output folder (keep the subfolder structure from path_to_models)
            video_a_relative_to_output = Path(video_a).relative_to(path_to_models)
            if is_baseline:
                video_b_relative_to_output = Path(video_b).relative_to(path_to_baslines)
            else:
                video_b_relative_to_output = Path(video_b).relative_to(path_to_models)

            video_a_output = output_folder / video_a_relative_to_output
            video_b_output = output_folder / video_b_relative_to_output

            video_a_output.parent.mkdir(parents=True, exist_ok=True)
            video_b_output.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy(video_a, video_a_output)
            shutil.copy(video_b, video_b_output)

            video_a_relative_to_output = Path(video_a_output).relative_to(study_folder)
            video_b_relative_to_output = Path(video_b_output).relative_to(study_folder)

            # # create video_a_output that is mute and video_b_output that is also mute
            # video_a_output_mute = video_a_output.parent / (video_a_output.stem + "_mute.mp4")
            # video_b_output_mute = video_b_output.parent / (video_b_output.stem + "_mute.mp4")

            # # use ffmpeg to mute the videos
            # os.system(f"ffmpeg -n -i {video_a_output} -vcodec copy -an {video_a_output_mute}")
            # os.system(f"ffmpeg -n -i {video_b_output} -vcodec copy -an {video_b_output_mute}")

            # video_a_output_mute_relative_to_output = Path(video_a_output_mute).relative_to(study_folder)
            # video_b_output_mute_relative_to_output = Path(video_b_output_mute).relative_to(study_folder)

            # intensity = video_a.parts[-2].split("_")[2]
            # intensity_b = video_b.parts[-2].split("_")[2]
            # assert intensity == intensity_b, f"Intensities do not match: {intensity} vs {intensity_b}"

            # randomly flip 50% of the videos
            flip = random.random() > 0.5
            
            flips[ri] += [flip]
            # selected_videos_a_emo[ri] += [video_a]
            # selected_videos_b_emo[ri] += [video_b]
            selected_videos_a_lip[ri] += [video_a]
            selected_videos_b_lip[ri] += [video_b]

            # hit list line has the following format: videopath_left#videopath_right#emotion#angry;videopath_left#videopath_right#lipsync#happy
            video_a = str(video_a)
            video_b = str(video_b)

            video_a_relative_to_output_dbg =  "http://0.0.0.0:8000/" + str(video_a_relative_to_output)
            video_b_relative_to_output_dbg =  "http://0.0.0.0:8000/" + str(video_b_relative_to_output)
            # video_a_output_mute_relative_to_output_dbg =   "http://0.0.0.0:8000/" + str(video_a_output_mute_relative_to_output)
            # video_b_output_mute_relative_to_output_dbg =   "http://0.0.0.0:8000/" + str(video_b_output_mute_relative_to_output)
            video_a_mturk = bucket_prefix  + str(video_a_relative_to_output)
            video_b_mturk = bucket_prefix  + str(video_b_relative_to_output)
            # video_a_output_mute_mturk = bucket_prefix + str(video_a_output_mute_relative_to_output)
            # video_b_output_mute_mturk = bucket_prefix +  str(video_b_output_mute_relative_to_output)

            if not flip:
                hit_list_line += [f"{video_a}#{video_b}"]
                hit_list_line_rel += [f"{video_a_relative_to_output}#{video_b_relative_to_output}"]
                hit_list_line_dbg += [f"{video_a_relative_to_output_dbg}#{video_b_relative_to_output_dbg}"]
                hit_list_line_mturk += [f"{video_a_mturk}#{video_b_mturk}"]
            else:
                hit_list_line += [f"{video_b}#{video_a}"]
                hit_list_line_rel += [f"{video_b_relative_to_output}#{video_a_relative_to_output}"]
                hit_list_line_dbg += [f"{video_b_relative_to_output_dbg}#{video_a_relative_to_output_dbg}"]
                hit_list_line_mturk += [f"{video_b_mturk}#{video_a_mturk}"]


            catch_trials[ri] += [0]
            video_count += 1
            i += 1

        # add catch trials
        for i in range(num_catch_trials):
            flip = random.random() > 0.5
            catch_trials[ri] += [1]
            flips[ri] += [flip]

            random_index = random.randint(0, len(catch_videos_lip_correct)-1)

            video_a_lip = catch_videos_lip_correct[random_index]
            video_b_lip = catch_videos_lip_wrong[random_index]
            
            video_a_lip_out = output_folder / video_a_lip.relative_to(lip_catch_path)
            video_b_lip_out = output_folder / video_b_lip.relative_to(lip_catch_path)

            shutil.copy(video_a_lip, output_folder / video_a_lip_out)
            shutil.copy(video_b_lip, output_folder / video_b_lip_out)

            video_a_lip_relative_to_output = Path(video_a_lip_out).relative_to(study_folder)
            video_b_lip_relative_to_output = Path(video_b_lip_out).relative_to(study_folder)

            selected_videos_a_lip[ri] += [video_a_lip]
            selected_videos_b_lip[ri] += [video_b_lip]

            video_a_lip = str(video_a_lip)
            video_b_lip = str(video_b_lip)
        
            video_a_lip_relative_to_output = str(video_a_lip_relative_to_output)
            video_b_lip_relative_to_output = str(video_b_lip_relative_to_output)

            video_a_lip_dbg = "http://0.0.0.0:8000/" + video_a_lip_relative_to_output
            video_b_lip_dbg = "http://0.0.0.0:8000/" + video_b_lip_relative_to_output

            video_a_lip_mturk = bucket_prefix + str(video_a_lip_relative_to_output)
            video_b_lip_mturk = bucket_prefix + str(video_b_lip_relative_to_output)

            if not flip:
                hit_list_line += [f"{video_a_lip}#{video_b_lip}"]
                hit_list_line_rel += [f"{video_a_lip_relative_to_output}#{video_b_lip_relative_to_output}"]
                hit_list_line_dbg += [f"{video_a_lip_dbg}#{video_b_lip_dbg}"]
                hit_list_line_mturk += [f"{video_a_lip_mturk}#{video_b_lip_mturk}"]
            else:
                hit_list_line += [f"{video_b_lip}#{video_a_lip}"]
                hit_list_line_rel += [f"{video_b_lip_relative_to_output}#{video_a_lip_relative_to_output}"]
                hit_list_line_dbg += [f"{video_b_lip_dbg}#{video_a_lip_dbg}"]
                hit_list_line_mturk += [f"{video_b_lip_mturk}#{video_a_lip_mturk}"]

        assert len(selected_videos_a_lip[ri]) == len(selected_videos_b_lip[ri]) == len(flips[ri]) == len(catch_trials[ri])
        

        # shuffle to mix in catch trials
        index_list = list(range(len(flips[ri])))
        random.shuffle(index_list)
        flips[ri] = [flips[ri][i] for i in index_list]
        selected_videos_a_lip[ri] = [str(selected_videos_a_lip[ri][i]) for i in index_list]
        selected_videos_b_lip[ri] = [str(selected_videos_b_lip[ri][i]) for i in index_list]
        catch_trials[ri] = [catch_trials[ri][i] for i in index_list]
        hit_list_line = [hit_list_line[i] for i in index_list]
        hit_list_line_rel = [hit_list_line_rel[i] for i in index_list]
        hit_list_line_dbg = [hit_list_line_dbg[i] for i in index_list]
        hit_list_line_mturk = [hit_list_line_mturk[i] for i in index_list]

        # repeat the first num_repeats elements at the end of the list
        flips[ri] += flips[ri][:num_repeats]
        selected_videos_a_lip[ri] += selected_videos_a_lip[ri][:num_repeats]
        selected_videos_b_lip[ri] += selected_videos_b_lip[ri][:num_repeats]
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
        videos_a_all = [str(video) for video in videos_a_all]
        videos_b_all = [str(video) for video in videos_b_all]

        lip_catch_vids = [str(video) for video in catch_videos_lip_correct + catch_videos_lip_wrong]
        
        for rvi in range(len(selected_videos_a_lip[ri])-num_repeats):
            video_to_rm = selected_videos_a_lip[ri][rvi]
            if video_to_rm in lip_catch_vids:
                continue
            videos_a_all.remove(video_to_rm)

        for rvi in range(len(selected_videos_b_lip[ri])-num_repeats):
            video_to_rm = selected_videos_b_lip[ri][rvi]
            if video_to_rm in lip_catch_vids:
                continue
            videos_b_all.remove(video_to_rm)

        # convert videos back to paths
        videos_a_all = [Path(video) for video in videos_a_all]
        videos_b_all = [Path(video) for video in videos_b_all]

    assert len(selected_videos_a_lip) == len(selected_videos_b_lip) == len(flips) == len(catch_trials)
    protocol_dict = {
        "model_a": model_a,
        "model_b": model_b,
        "flips": flips, 
        "videos_a_lip": selected_videos_a_lip,
        "videos_b_lip": selected_videos_b_lip,
        "catch_trials": catch_trials,
        "num_repeats": num_repeats,
        "num_catch_trials": num_catch_trials,
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
    # ## final ENSPARC model (WITH prior, lip reading, video emotion, disentanglement) - to be revised
    # main_model = "2023_05_08_20-36-09_8797431074914794141_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
    # main_model_videos_folder = Path(path_to_models) / main_model / video_folder
    
    ## NEW MAIN MODEL 
    main_model = "2023_05_18_01-26-32_-6224330163499889169_FaceFormer_MEADP_Awav2vec2_Elinear_DBertPriorDecoder_Seml_NPE_Tff_predEJ_LVmmmLmm"
    main_model_videos_folder = Path(path_to_models) / main_model / video_folder
    # main_model_videos = read_rendered_vids(main_model_videos_folder)
    main_model_videos = search_neutral_rendered_vids(main_model_videos_folder,  pattern="M003")

    # ablation models
    baselines_models = []
    ## ENSPARC with prior but no perceptual
    suffix = ""
    if lrs_subset == "test":
        suffix = "_test"
        mt_suffix = suffix
    else:
        mt_suffix = "_new"

    is_baseline = []

    baselines_models += [f"CT{suffix}_reorg"] 
    is_baseline += [True]
    baselines_models += [f"FF{suffix}_reorg"] 
    is_baseline += [True]
    baselines_models += [f"VOCA{suffix}_reorg"] 
    is_baseline += [True]
    baselines_models += [f"MT{mt_suffix}_reorg"] 
    is_baseline += [True]
    baselines_models += ["2023_05_10_13-16-00_-3885098104460673227_FaceFormer_MEADP_Awav2vec2T_Elinear_DFaceFormerDecoder_Seml_PPE_predV_LV"]
    is_baseline += [False]

    ablation_vids = []

    for bi, model_b in enumerate(baselines_models):
        if is_baseline[bi]:
            vids = replace_rendered_baseline_model(main_model_videos, model_b)
        else:
            vids = replace_rendered_vid_model(main_model_videos, model_b, model_idx=8)
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
    videos_per_row = 15
    repeats = 3
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
    for mi, model_b in enumerate(baselines_models): 
        output_folder = Path(path_to_models) / model_b / "mturk"
        output_folder.mkdir(parents=True, exist_ok=True)
        output_folder = Path(path_to_studies) / study_name / f"main_vs_{mi}"
        hitlist = design_study_3(main_model, model_b, num_rows, videos_per_row, output_folder, num_catch_trials=num_catch_trials, videos_a=main_model_videos, num_repeats=repeats, is_baseline=is_baseline[mi])
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
