How to release, in case you forget:

git checkout master
git pull to make sure you have all changes
update HISTORY.rst with the new version
bumpversion [major/minor/patch] will increase version, commit, and make a new tag
git push origin <tagname> tags must be manually pushed to github
python setup.py sdist upload -r pypi
Add a new "Latest" header to HISTORY.rst, commit, and push
