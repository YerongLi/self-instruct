awk 'NR % 4 == 0 { print > "police1.json" } NR % 4 != 0 { print > "police.json" }' police-full.json
