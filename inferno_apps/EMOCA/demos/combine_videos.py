from pathlib import Path 
import os, sys 



def main(): 
    # input_folder1 = Path("/ps/scratch/rdanecek/EMOCA/lrs3")
    # input_folder2 = Path("/ps/scratch/rdanecek/Deep3DFace/lrs3")
    # output_folder = Path("/ps/scratch/rdanecek/EMOCA_Deep3dFace_videos/lrs3")

    # input_folder1 = Path("/ps/scratch/rdanecek/EMOCA/lrs3_nodetect")
    # input_folder2 = Path("/ps/scratch/rdanecek/Deep3DFace/lrs3_nodetect")
    # output_folder = Path("/ps/scratch/rdanecek/EMOCA_Deep3dFace_videos/lrs3_nodetect")

    # input_folder1 = Path("/ps/scratch/rdanecek/EMOCA/lrs3_3fabrec")
    # input_folder2 = Path("/ps/scratch/rdanecek/Deep3DFace/lrs3_3fabrec")
    # output_folder = Path("/ps/scratch/rdanecek/EMOCA_Deep3dFace_videos/lrs3_3fabrec")


    input_folder1 = Path("/ps/scratch/rdanecek/EMOCA_Deep3dFace_videos/lrs3")
    input_folder2 = Path("/ps/scratch/rdanecek/EMOCA_Deep3dFace_videos/lrs3_3fabrec")
    output_folder = Path("/ps/scratch/rdanecek/EMOCA_Deep3dFace_videos_comp/lrs3_3fabrec")

    # glob for videos 
    video_paths1 = sorted(input_folder1.glob("*.mp4"))
    video_paths2 = sorted(input_folder2.glob("*.mp4"))

    # get a set of filenames 
    video_filenames1 = set([p.name for p in video_paths1])
    video_filenames2 = set([p.name for p in video_paths2]) 

    # get a set of filenames that are in both folders
    common_filenames = video_filenames1.intersection(video_filenames2)

    output_folder.mkdir(exist_ok=True, parents=True)

    for fname in common_filenames: 
        video_fname1 = input_folder1 / fname
        video_fname2 = input_folder2 / fname

        # get the output filename
        output_fname = output_folder / fname

        # # stack the videos horizontally
        # cmd = "ffmpeg -i {} -i {} -filter_complex hstack {}".format(video_fname1, video_fname2, output_fname)

        # stack the videos horizontally and keep the audio from the first video
        # cmd = "ffmpeg -i {} -i {} -filter_complex hstack -map 0:0 -map 1:0 {}".format(video_fname1, video_fname2, output_fname)
        
        # cmd = "ffmpeg -i {} -i {} -filter_complex hstack {}".format(video_fname1, video_fname2, output_fname)

        ## create a side by side video from the two videos using ffmpg, take the audio from the first video
        cmd = f"ffmpeg -i {video_fname1} -i {video_fname2}  -filter_complex hstack -filter_complex \"[0:a][1:a]amerge\" -map 0:v -map 1:v {output_fname}"

        os.system(cmd)

        print("Created: ", output_fname)



if __name__ == "__main__": 
    main()
