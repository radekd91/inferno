for file in "$1"/*
do
#   echo $file
   f=$(basename --  "$file")
   ffmpeg -i "$1"/"$f" -r 25/1 -c:v h264 -q:v 1 "$2"/"$f"
   # echo "$1"/"$f"
   # echo "$2"/"$f"
#   echo "$f"
done

