rm final_submission.zip
mkdir final_submission/
mkdir final_submission/src/

cp -r src/utils final_submission/src/
cp -r src/simulation_attributes final_submission/src/
cp -r src/experiments.py final_submission/src/

cp -r media/ final_submission/
cp -r run_scripts/ final_submission/
cp README.md final_submission/
cp requirements.txt final_submission/
cp run_experiment.py final_submission/

cp -r submission/submission.pdf final_submission/
cp -r submission/out/submission.pdf final_submission/

echo "### Create ZIP ###"

zip final_submission -r final_submission/

rm -r final_submission/