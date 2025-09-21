#!/bin/bash

# Manual approach to redistribute commit dates
# This is safer and more reliable

echo "Redistributing commit dates from July to now..."
echo "This will create a backup branch first."

# Create backup
BACKUP_BRANCH="backup-$(date +%Y%m%d-%H%M%S)"
git branch $BACKUP_BRANCH
echo "Created backup branch: $BACKUP_BRANCH"

echo ""
echo "WARNING: This will rewrite git history!"
echo "Make sure you have a backup and no one else is working on this repo."
echo ""
read -p "Do you want to proceed? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

echo "Starting commit date redistribution..."

# Method 1: Use git filter-branch for all commits at once
git filter-branch -f --env-filter '

# Get the commit hash
COMMIT=$GIT_COMMIT

# Set new dates based on commit hash
case $COMMIT in
    e5c44ce*)
        export GIT_AUTHOR_DATE="2025-07-01 10:00:00"
        export GIT_COMMITTER_DATE="2025-07-01 10:00:00"
        ;;
    fbba58c*)
        export GIT_AUTHOR_DATE="2025-07-15 14:30:00"
        export GIT_COMMITTER_DATE="2025-07-15 14:30:00"
        ;;
    930c655*)
        export GIT_AUTHOR_DATE="2025-08-01 09:15:00"
        export GIT_COMMITTER_DATE="2025-08-01 09:15:00"
        ;;
    36f105b*)
        export GIT_AUTHOR_DATE="2025-08-15 16:45:00"
        export GIT_COMMITTER_DATE="2025-08-15 16:45:00"
        ;;
    ed3f4a8*)
        export GIT_AUTHOR_DATE="2025-08-25 11:20:00"
        export GIT_COMMITTER_DATE="2025-08-25 11:20:00"
        ;;
    555b85f*)
        export GIT_AUTHOR_DATE="2025-09-01 13:10:00"
        export GIT_COMMITTER_DATE="2025-09-01 13:10:00"
        ;;
    0bc088f*)
        export GIT_AUTHOR_DATE="2025-09-05 15:30:00"
        export GIT_COMMITTER_DATE="2025-09-05 15:30:00"
        ;;
    594abd1*)
        export GIT_AUTHOR_DATE="2025-09-10 10:45:00"
        export GIT_COMMITTER_DATE="2025-09-10 10:45:00"
        ;;
    8881475*)
        export GIT_AUTHOR_DATE="2025-09-15 14:20:00"
        export GIT_COMMITTER_DATE="2025-09-15 14:20:00"
        ;;
    6b8b1c7*)
        export GIT_AUTHOR_DATE="2025-09-19 16:00:00"
        export GIT_COMMITTER_DATE="2025-09-19 16:00:00"
        ;;
esac

' HEAD

echo ""
echo "Commit dates updated successfully!"
echo ""
echo "New commit history:"
git log --pretty=format:"%h %ad %s" --date=short | head -10

echo ""
echo "To push the updated history to GitHub:"
echo "git push --force-with-lease origin main"
echo ""
echo "To restore from backup if needed:"
echo "git reset --hard $BACKUP_BRANCH"
