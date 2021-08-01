rm -r final_submission/
rm final_submission.zip
mkdir final_submission/
rsync -av log/ final_submission/log/ --exclude "*.npy"
cp -r media/ final_submission/
cp -r run_scripts/ final_submission/
cp -r src/ final_submission/
cp -r submission/ final_submission/
cp -r test/ final_submission/
cp README.md final_submission/
cp requirements.txt final_submission/
cp run_experiment.py final_submission/
rm -r final_submission/submission/out final_submission/submission/memo.tex

echo "### Create ZIP ###"

zip final_submission -r final_submission/