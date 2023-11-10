# find "Demonstration Videos" -name "*.mp4" -print0 | xargs -0 -n 1 -P 12 ./run.sh
# Set the maximum number of parallel processes
MAX_PROCESSES=12

# Find all .mp4 files in the "Demonstration Videos" directory
find "Demonstration Videos" -name "*.mp4" | while read -r file; do
    # Run the ./run.sh script in the background
    ./run.sh "$file" &

    # Check the number of background processes and wait if it reaches the limit
    while [ $(jobs | wc -l) -ge $MAX_PROCESSES ]; do
        sleep 1
    done
done

# Wait for all background processes to finish
wait
