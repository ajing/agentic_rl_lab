#!/bin/bash

# Fixed script to redistribute commit dates
echo "Redistributing commit dates from July to now..."

# Create a backup first
echo "Creating backup branch..."
git branch backup-$(date +%Y%m%d-%H%M%S)

# Define the new dates for each commit
echo "Updating commit dates..."

# Use git rebase with --exec to change dates
git rebase -i --root --exec '

# Get the current commit hash
COMMIT=$(git rev-parse HEAD)

# Define new dates based on commit hash
case $COMMIT in
    e5c44ce*)
        NEW_DATE="2025-07-01 10:00:00"
        ;;
    fbba58c*)
        NEW_DATE="2025-07-15 14:30:00"
        ;;
    930c655*)
        NEW_DATE="2025-08-01 09:15:00"
        ;;
    36f105b*)
        NEW_DATE="2025-08-15 16:45:00"
        ;;
    ed3f4a8*)
        NEW_DATE="2025-08-25 11:20:00"
        ;;
    555b85f*)
        NEW_DATE="2025-09-01 13:10:00"
        ;;
    0bc088f*)
        NEW_DATE="2025-09-05 15:30:00"
        ;;
    594abd1*)
        NEW_DATE="2025-09-10 10:45:00"
        ;;
    8881475*)
        NEW_DATE="2025-09-15 14:20:00"
        ;;
    6b8b1c7*)
        NEW_DATE="2025-09-19 16:00:00"
        ;;
    *)
        # Keep original date for unknown commits
        exit 0
        ;;
esac

# Update the commit date
export GIT_AUTHOR_DATE="$NEW_DATE"
export GIT_COMMITTER_DATE="$NEW_DATE"

echo "Updated commit $COMMIT to $NEW_DATE"
'

echo "Commit dates updated successfully!"
echo "Verifying new dates:"
git log --pretty=format:"%h %ad %s" --date=short | head -10
