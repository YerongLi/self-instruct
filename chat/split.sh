awk 'NR % 4 == 0 { print > "police1.json" } NR % 4 != 0 { print > "police.json" }' input_file.json
