# find "Demonstration Videos" -name "*.mp4" -print0 | xargs -0 -n 1 -P 12 ./run.sh
# Set the maximum number of parallel processes
MAX_PROCESSES=12
COMMAND="python demo/topdown_demo_with_mmdet.py"

find "Demonstration Videos" -name "*.mp4" | while read -r file; do
    ./run.sh "$file" &

    while [ $(jobs | grep "$COMMAND" | wc -l) -ge $MAX_PROCESSES ]; do
        sleep 1
    done
done

wait
