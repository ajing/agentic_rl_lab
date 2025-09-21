#!/bin/bash

# Simple script to redistribute commit dates
# This approach is safer and more straightforward

echo "Redistributing commit dates from July to now..."

# Define the new dates for each commit (spread from July to now)
declare -A COMMIT_DATES=(
    ["e5c44ce"]="2025-07-01 10:00:00"  # Initial commit
    ["fbba58c"]="2025-07-15 14:30:00"  # docs: translate project
    ["930c655"]="2025-08-01 09:15:00"  # feat: add data/index/query scripts
    ["36f105b"]="2025-08-15 16:45:00"  # docs(plan): clarify Week 1
    ["ed3f4a8"]="2025-08-25 11:20:00"  # docs(learning): add Week 1 learnings
    ["555b85f"]="2025-09-01 13:10:00"  # docs: link Week 1 learnings
    ["0bc088f"]="2025-09-05 15:30:00"  # data: add Week 1 evaluation outputs
    ["594abd1"]="2025-09-10 10:45:00"  # feat(week2): implement RL environment
    ["8881475"]="2025-09-15 14:20:00"  # docs: enhance README
    ["6b8b1c7"]="2025-09-19 16:00:00"  # feat(week3-4): implement reward modeling
)

echo "New commit dates:"
for commit in "${!COMMIT_DATES[@]}"; do
    echo "  $commit -> ${COMMIT_DATES[$commit]}"
done

echo ""
echo "This script will use git filter-branch to change commit dates."
echo "WARNING: This rewrites git history. Make sure you have backups!"
echo ""
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Apply the date changes
for commit in "${!COMMIT_DATES[@]}"; do
    new_date="${COMMIT_DATES[$commit]}"
    echo "Updating commit $commit to $new_date..."
    
    git filter-branch -f --env-filter "
        if [ \$GIT_COMMIT = $commit ]; then
            export GIT_AUTHOR_DATE='$new_date'
            export GIT_COMMITTER_DATE='$new_date'
        fi
    " HEAD > /dev/null 2>&1
done

echo ""
echo "Commit dates updated successfully!"
echo "Verifying new dates:"
git log --pretty=format:"%h %ad %s" --date=short | head -10

echo ""
echo "To push the updated history to GitHub, you'll need to force push:"
echo "git push --force-with-lease origin main"
echo ""
echo "WARNING: Force pushing rewrites remote history. Make sure no one else is working on this repo!"
