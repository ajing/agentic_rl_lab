#!/bin/bash

# Script to redistribute commit dates from July to now
# This will spread out commits more naturally over time

echo "Redistributing commit dates from July to now..."

# Get the list of commits (excluding the initial commit)
COMMITS=$(git log --oneline --reverse | tail -n +2 | awk '{print $1}')

# Define date range: July 1, 2025 to today
START_DATE="2025-07-01"
END_DATE="2025-09-19"

# Convert to seconds since epoch
START_EPOCH=$(date -j -f "%Y-%m-%d" "$START_DATE" "+%s")
END_EPOCH=$(date -j -f "%Y-%m-%d" "$END_DATE" "+%s")

# Calculate total range in seconds
RANGE=$((END_EPOCH - START_EPOCH))

# Count commits
COMMIT_COUNT=$(echo "$COMMITS" | wc -l | tr -d ' ')

echo "Found $COMMIT_COUNT commits to redistribute"
echo "Date range: $START_DATE to $END_DATE"

# Create array of commit hashes
COMMIT_ARRAY=($COMMITS)

# Redistribute commits
for i in "${!COMMIT_ARRAY[@]}"; do
    COMMIT_HASH="${COMMIT_ARRAY[$i]}"
    
    # Calculate new date (spread evenly across the range)
    if [ $COMMIT_COUNT -eq 1 ]; then
        NEW_EPOCH=$START_EPOCH
    else
        INTERVAL=$((RANGE / (COMMIT_COUNT - 1)))
        NEW_EPOCH=$((START_EPOCH + i * INTERVAL))
    fi
    
    # Convert back to date format
    NEW_DATE=$(date -r $NEW_EPOCH "+%Y-%m-%d %H:%M:%S")
    
    echo "Commit $((i+1))/$COMMIT_COUNT: $COMMIT_HASH -> $NEW_DATE"
    
    # Use git filter-branch to change the commit date
    git filter-branch -f --env-filter "
        if [ \$GIT_COMMIT = $COMMIT_HASH ]; then
            export GIT_AUTHOR_DATE='$NEW_DATE'
            export GIT_COMMITTER_DATE='$NEW_DATE'
        fi
    " HEAD~$((COMMIT_COUNT - i))..HEAD > /dev/null 2>&1
done

echo "Commit dates redistributed successfully!"
echo "Run 'git log --pretty=format:\"%h %ad %s\" --date=short' to verify"
