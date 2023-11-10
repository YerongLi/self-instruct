task=$1
find "Demonstration Videos" -name "*.mp4" -print0 | xargs -0 -n 1 -P 12 ./run_$task.sh

