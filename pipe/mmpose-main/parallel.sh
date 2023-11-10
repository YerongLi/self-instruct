cd "Demonstration Videos"

# Step 5: Iterate over each .mp4 file and run the command
find . -name "*.mp4" -print0 | xargs -0 -n 1 -P 8 ./run.sh
